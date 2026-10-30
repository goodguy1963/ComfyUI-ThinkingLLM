"""Native OpenAI Decisions and OpenRouter Decisions API adapters."""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping


OPENAI_DECISION_MODEL = "gpt-6-luna"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
MODEL_CACHE_TTL_SECONDS = 3600
_MODEL_CACHE: dict[str, tuple[float, list[dict]]] = {}


def _request_json(url: str, *, headers: dict[str, str], payload=None, timeout: int = 60, opener=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {"Accept": "application/json", **headers}
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=request_headers, method="GET" if data is None else "POST")
    try:
        response = (opener or urllib.request.build_opener()).open(req, timeout=timeout)
        with response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        # Do not include provider response text: it can echo user content or media.
        raise RuntimeError(f"Decision API returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach the Decision API ({type(exc.reason).__name__}).") from exc
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Decision API returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Decision API returned an unexpected response shape.")
    if result.get("error"):
        raise RuntimeError("Decision API reported an inference error.")
    return result


def _route_criteria(routes):
    return {str(route["name"]): str(route.get("description") or "") or None for route in routes}


def build_decision_request(protocol: str, model: str, input_kind: str, item, image_data_url: str | None, system_prompt: str, routes):
    criteria = _route_criteria(routes)
    instruction = str(system_prompt or "").strip() or "Choose the single route that best matches the input."
    if protocol == "openai_decisions":
        content = [{"type": "input_text", "text": str(item) if input_kind == "text" else "Classify the supplied image."}]
        if image_data_url:
            content.append({"type": "input_image", "image_url": image_data_url})
        return {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "questions": [{
                "type": "choice",
                "name": "route",
                "instructions": instruction,
                "choices": [{"value": name, "description": description or ""} for name, description in criteria.items()],
            }],
        }
    if protocol == "openrouter_decisions":
        state = str(item) if input_kind == "text" else []
        if input_kind == "image":
            state = [{"type": "image_url", "image_url": {"url": image_data_url}}]
        return {
            "model": model,
            "state": state,
            "questions": {
                "route": {"type": "choice", "instructions": instruction, "criteria": criteria},
            },
        }
    raise ValueError(f"Unsupported Decision API protocol: {protocol!r}.")


def _probability_map(raw, route_names):
    if isinstance(raw, Mapping):
        values = {str(key): value for key, value in raw.items()}
    elif isinstance(raw, list):
        values = {}
        for item in raw:
            if not isinstance(item, dict) or "value" not in item:
                raise RuntimeError("Decision API returned malformed route probabilities.")
            value = str(item["value"])
            if value in values:
                raise RuntimeError("Decision API returned duplicate route probabilities.")
            values[value] = item.get("probability")
    else:
        raise RuntimeError("Decision API returned no route probabilities.")
    if set(values) != set(route_names):
        raise RuntimeError("Decision API probabilities do not match the configured route names.")
    normalized = {}
    for name in route_names:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise RuntimeError("Decision API returned an invalid route probability.")
        normalized[name] = float(value)
    return normalized


def normalize_decision_response(protocol: str, response: dict, route_names: list[str]):
    if protocol == "openai_decisions":
        answers = response.get("answers")
        answer = next((item for item in answers if isinstance(item, dict) and item.get("name") == "route"), None) if isinstance(answers, list) else None
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise RuntimeError("OpenAI Decisions API did not return a route choice.")
        choice = answer.get("choice")
        probability_map = _probability_map(answer.get("probabilities"), route_names)
        confidence = answer.get("confidence")
    elif protocol == "openrouter_decisions":
        answers = response.get("answers")
        answer = answers.get("route") if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise RuntimeError("OpenRouter Decisions API did not return a route choice.")
        choice = answer.get("choice")
        probability_map = _probability_map(answer.get("probabilities"), route_names)
        confidence = answer.get("confidence")
    else:
        raise ValueError(f"Unsupported Decision API protocol: {protocol!r}.")
    if not isinstance(choice, str) or choice not in route_names:
        raise RuntimeError("Decision API selected an unknown route.")
    if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1):
        raise RuntimeError("Decision API returned an invalid confidence value.")
    return choice, probability_map, None if confidence is None else float(confidence)


def request_decision(profile: dict, model: str, input_kind: str, item, image_data_url: str | None, system_prompt: str, routes, *, api_key: str, opener=None):
    protocol = profile.get("decision_protocol")
    payload = build_decision_request(protocol, model, input_kind, item, image_data_url, system_prompt, routes)
    response = _request_json(
        profile["decision_endpoint"],
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        payload=payload,
        timeout=int(profile.get("max_timeout_seconds", 300)),
        opener=opener,
    )
    return normalize_decision_response(protocol, response, [str(route["name"]) for route in routes])


def build_fields_decision_request(
    protocol: str,
    model: str,
    input_kind: str,
    item,
    image_data_url: str | None,
    system_prompt: str,
    fields,
):
    """Build one provider request containing all typed output fields for an input."""
    instructions = str(system_prompt or "").strip()
    questions = []
    openrouter_questions = {}
    for field in fields:
        question_text = str(field["question"])
        instruction = f"{instructions}\n\n{question_text}" if instructions else question_text
        options = field["options"]
        if protocol == "openai_decisions":
            questions.append({
                "type": "choice",
                "name": str(field["id"]),
                "instructions": instruction,
                "choices": [
                    {"value": str(option["value"]), "description": str(option["description"])}
                    for option in options
                ],
            })
        elif protocol == "openrouter_decisions":
            openrouter_questions[str(field["id"])] = {
                "type": "choice",
                "instructions": instruction,
                "criteria": {
                    str(option["value"]): str(option["description"])
                    for option in options
                },
            }
        else:
            raise ValueError(f"Unsupported Decision API protocol: {protocol!r}.")

    if protocol == "openai_decisions":
        content = [{
            "type": "input_text",
            "text": str(item) if input_kind == "text" else "Evaluate the configured fields for the supplied image.",
        }]
        if image_data_url:
            content.append({"type": "input_image", "image_url": image_data_url})
        return {
            "model": model,
            "input": [{"role": "user", "content": content}],
            "questions": questions,
        }

    state = str(item) if input_kind == "text" else []
    if input_kind == "image":
        state = [{"type": "image_url", "image_url": {"url": image_data_url}}]
    return {"model": model, "state": state, "questions": openrouter_questions}


def normalize_fields_decision_response(protocol: str, response: dict, fields):
    """Normalize the differently shaped provider responses by stable field id."""
    expected_ids = {str(field["id"]) for field in fields}
    if protocol == "openai_decisions":
        raw_answers = response.get("answers")
        if not isinstance(raw_answers, list):
            raise RuntimeError("OpenAI Decisions API did not return field answers.")
        answers = {}
        for answer in raw_answers:
            if not isinstance(answer, dict) or answer.get("name") not in expected_ids:
                continue
            field_id = str(answer["name"])
            if field_id in answers:
                raise RuntimeError(f"OpenAI Decisions API returned duplicate answers for {field_id}.")
            answers[field_id] = answer
    elif protocol == "openrouter_decisions":
        raw_answers = response.get("answers")
        if not isinstance(raw_answers, dict):
            raise RuntimeError("OpenRouter Decisions API did not return field answers.")
        answers = raw_answers
    else:
        raise ValueError(f"Unsupported Decision API protocol: {protocol!r}.")

    normalized = {}
    for field in fields:
        field_id = str(field["id"])
        answer = answers.get(field_id)
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise RuntimeError(f"Decision API did not return a choice for field {field['name']!r}.")
        option_values = [str(option["value"]) for option in field["options"]]
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in option_values:
            raise RuntimeError(f"Decision API selected an unknown value for field {field['name']!r}.")
        probabilities = _probability_map(answer.get("probabilities"), option_values)
        confidence = answer.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise RuntimeError(f"Decision API returned invalid confidence for field {field['name']!r}.")
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise RuntimeError(f"Decision API returned invalid confidence for field {field['name']!r}.")
        normalized[field_id] = (choice, probabilities, confidence)
    return normalized


def request_decision_fields(
    profile: dict,
    model: str,
    input_kind: str,
    item,
    image_data_url: str | None,
    system_prompt: str,
    fields,
    *,
    api_key: str,
    opener=None,
):
    """Ask one API request to evaluate all configured output fields."""
    protocol = profile.get("decision_protocol")
    payload = build_fields_decision_request(
        protocol, model, input_kind, item, image_data_url, system_prompt, fields,
    )
    response = _request_json(
        profile["decision_endpoint"],
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        payload=payload,
        timeout=int(profile.get("max_timeout_seconds", 300)),
        opener=opener,
    )
    return normalize_fields_decision_response(protocol, response, fields)


def openrouter_decision_models(input_kind: str, *, timeout: int = 20, now: float | None = None, opener=None):
    """Return OpenRouter model ids with decision output and requested input capability."""
    modality = "image" if input_kind == "image" else "text"
    now = time.monotonic() if now is None else now
    cached = _MODEL_CACHE.get(modality)
    if cached and now - cached[0] < MODEL_CACHE_TTL_SECONDS:
        return [dict(model) for model in cached[1]]
    query = urllib.parse.urlencode({"output_modalities": "decisions", "input_modalities": modality})
    try:
        data = _request_json(f"{OPENROUTER_MODELS_URL}?{query}", headers={}, timeout=timeout, opener=opener)
        items = data.get("data")
        if not isinstance(items, list):
            raise RuntimeError("OpenRouter model catalog returned an unexpected response shape.")
        models = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            architecture = item.get("architecture")
            modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else None
            output_modalities = architecture.get("output_modalities") if isinstance(architecture, dict) else None
            if not isinstance(modalities, list) or modality not in modalities:
                continue
            if not isinstance(output_modalities, list) or "decisions" not in output_modalities:
                continue
            models.append({"id": item["id"], "label": item.get("name") or item["id"], "vision": "image" in modalities})
        if not models:
            raise RuntimeError("OpenRouter returned no decision models for the requested input type.")
        _MODEL_CACHE[modality] = (now, models)
        return [dict(model) for model in models]
    except RuntimeError:
        if cached:
            return [dict(model) for model in cached[1]]
        raise


def decision_models_for_profile(profile: dict, input_kind: str):
    if input_kind == "image" and not profile.get("allow_images", False):
        return []
    protocol = profile.get("decision_protocol")
    if protocol == "openai_decisions":
        models = [{"id": OPENAI_DECISION_MODEL, "label": "GPT-6 Luna Decisions", "vision": True}]
    elif protocol == "openrouter_decisions":
        models = openrouter_decision_models(input_kind)
    else:
        return []
    allowed = profile.get("allowed_models")
    if allowed is not None:
        models = [model for model in models if model["id"] in allowed]
    return models
