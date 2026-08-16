"""OpenAI-compatible API node for ThinkingLLM.

Supports QwenCloud, OrcaRouter, Featherless, and arbitrary OpenAI-compatible
chat-completions endpoints without adding an SDK dependency.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from AILab_StreamDisplay import (
    compose_streamed_model_output,
    extract_stream_token,
    get_thinking_stream_display,
)


PROVIDERS = {
    "QwenCloud": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "OrcaRouter": {
        "base_url": "https://api.orcarouter.ai/v1",
        "api_key_env": "ORCAROUTER_API_KEY",
    },
    "Featherless": {
        "base_url": "https://api.featherless.ai/v1",
        "api_key_env": "FEATHERLESS_API_KEY",
    },
    "Custom": {
        "base_url": "",
        "api_key_env": "OPENAI_API_KEY",
    },
}


def _provider_config(provider: str) -> dict[str, str]:
    return PROVIDERS.get(provider, PROVIDERS["Custom"])


def _resolve_base_url(provider: str, base_url: str) -> str:
    value = str(base_url or "").strip()
    if not value:
        value = _provider_config(provider)["base_url"]
    if not value:
        raise ValueError("base_url is required when provider is Custom.")
    return value.rstrip("/")


def _resolve_api_key(provider: str, api_key_env: str) -> tuple[str, str]:
    env_name = str(api_key_env or "").strip() or _provider_config(provider)["api_key_env"]
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise RuntimeError(
            f"API key environment variable {env_name!r} is not set. "
            "Set it before starting ComfyUI so the key is not stored in the workflow."
        )
    return key, env_name


def _parse_extra_json(extra_body_json: str) -> dict:
    raw = str(extra_body_json or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"extra_body_json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("extra_body_json must contain a JSON object.")
    return value


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        data = exc.read()
    except Exception:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace").strip()
    return str(data or "").strip()


def _extract_nonstream_response(payload: dict) -> tuple[str, str]:
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content") or choice.get("text") or ""
    reasoning = (
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("thinking")
        or choice.get("reasoning_content")
        or ""
    )
    return str(content or ""), str(reasoning or "")


class ThinkingLLMOpenAICompatibleAPI:
    """Call a remote OpenAI-compatible chat-completions endpoint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (
                    list(PROVIDERS),
                    {
                        "default": "QwenCloud",
                        "tooltip": (
                            "Provider preset. QwenCloud, OrcaRouter and Featherless fill in "
                            "their standard base URL and API-key environment variable."
                        ),
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "qwen3.8-max-preview",
                        "multiline": False,
                        "tooltip": (
                            "Exact model ID exposed by the selected provider. "
                            "For community/uncensored models, paste the provider's model ID here."
                        ),
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are a helpful assistant.",
                        "multiline": True,
                        "tooltip": "Optional system message sent before the user prompt.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "User message sent to the model.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 1,
                        "max": 65536,
                        "tooltip": "Maximum number of output tokens requested from the API.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 2**32 - 1,
                    },
                ),
                "enable_thinking": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "For QwenCloud this sends enable_thinking. Other providers can "
                            "receive provider-specific reasoning flags through extra_body_json."
                        ),
                    },
                ),
                "thinking_budget": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 0,
                        "max": 262144,
                        "tooltip": "QwenCloud thinking budget. Set to 0 to omit the parameter.",
                    },
                ),
                "stream_tokens_to_terminal": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Print streamed reasoning/answer tokens to the ComfyUI terminal.",
                    },
                ),
                "timeout_seconds": (
                    "INT",
                    {
                        "default": 300,
                        "min": 10,
                        "max": 3600,
                    },
                ),
            },
            "optional": {
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Optional endpoint override. Leave empty to use the provider preset.",
                    },
                ),
                "api_key_env": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Environment variable containing the API key. Leave empty for the "
                            "provider default. The secret itself is never stored in the workflow."
                        ),
                    },
                ),
                "extra_body_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Optional JSON object merged into the request body for provider-specific "
                            "parameters, e.g. routing or reasoning options."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "RAW_TRACE")
    FUNCTION = "generate"
    CATEGORY = "ThinkingLLM/API"
    DESCRIPTION = (
        "Calls any OpenAI-compatible /chat/completions API and returns the final "
        "response plus a raw reasoning trace when exposed by the provider."
    )

    def generate(
        self,
        provider,
        model_name,
        system_prompt,
        prompt,
        max_tokens,
        temperature,
        top_p,
        seed,
        enable_thinking,
        thinking_budget,
        stream_tokens_to_terminal,
        timeout_seconds,
        base_url="",
        api_key_env="",
        extra_body_json="",
    ):
        model = str(model_name or "").strip()
        if not model:
            raise ValueError("model_name must not be empty.")

        endpoint = _resolve_base_url(provider, base_url) + "/chat/completions"
        api_key, env_name = _resolve_api_key(provider, api_key_env)

        messages = []
        if str(system_prompt or "").strip():
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(prompt or "")})

        request_body = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "seed": int(seed),
            "stream": True,
        }

        if provider == "QwenCloud":
            request_body["enable_thinking"] = bool(enable_thinking)
            if int(thinking_budget) > 0:
                request_body["thinking_budget"] = int(thinking_budget)

        extra = _parse_extra_json(extra_body_json)
        request_body.update(extra)
        request_body["stream"] = True

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": "ComfyUI-ThinkingLLM/OpenAI-Compatible-API",
            },
            method="POST",
        )

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        display = get_thinking_stream_display() if stream_tokens_to_terminal else None
        if display is not None:
            display.start_stage(f"API {provider} / {model}")

        try:
            with urllib.request.urlopen(req, timeout=int(timeout_seconds)) as response:
                content_type = str(response.headers.get("Content-Type", "")).lower()
                if "text/event-stream" not in content_type:
                    raw = response.read().decode("utf-8", errors="replace")
                    payload = json.loads(raw)
                    content, reasoning = _extract_nonstream_response(payload)
                    content_parts.append(content)
                    reasoning_parts.append(reasoning)
                    if display is not None:
                        display.push_compact(reasoning or content)
                else:
                    for raw_line in response:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        token = extract_stream_token(chunk)
                        reasoning = token.get("reasoning", "")
                        content = token.get("content", "")
                        if reasoning:
                            reasoning_parts.append(reasoning)
                            if display is not None:
                                display.push_compact(reasoning)
                        if content:
                            content_parts.append(content)
                            if display is not None:
                                display.push_compact(content)
        except urllib.error.HTTPError as exc:
            body = _read_error_body(exc)
            detail = f"HTTP {exc.code} {exc.reason}"
            if body:
                detail += f": {body}"
            raise RuntimeError(
                f"ThinkingLLM API request failed ({provider}, key env {env_name}): {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"ThinkingLLM API request failed ({provider}, {endpoint}): {exc.reason}"
            ) from exc
        finally:
            if display is not None:
                display.end_stage()

        content_text = "".join(content_parts).strip()
        reasoning_text = "".join(reasoning_parts).strip()
        raw_trace = compose_streamed_model_output(reasoning_text, content_text)

        return (content_text, raw_trace)


NODE_CLASS_MAPPINGS = {
    "ThinkingLLM_OpenAICompatibleAPI": ThinkingLLMOpenAICompatibleAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThinkingLLM_OpenAICompatibleAPI": "ThinkingLLM API (OpenAI Compatible)",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
