"""Secure OpenAI-compatible API node for ThinkingLLM.

The workflow never receives API keys, environment-variable names, or arbitrary
remote endpoints. Instead it selects a server-side API profile that binds the
endpoint and authentication policy together.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from AILab_OutputCleaner import OutputCleanConfig, clean_model_output
from AILab_StreamDisplay import (
    compose_streamed_model_output,
    extract_stream_token,
    get_thinking_stream_display,
)


MODULE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = MODULE_DIR.parent
DEFAULT_PROFILE_FILE = PACKAGE_DIR / "thinkingllm_api_profiles.json"
PROFILE_FILE_ENV = "THINKINGLLM_API_PROFILES_FILE"

_RESERVED_EXTRA_BODY_FIELDS = {
    "model",
    "messages",
    "stream",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "seed",
    "enable_thinking",
    "thinking_budget",
}
DISABLE_BUILTINS_ENV = "THINKINGLLM_DISABLE_BUILTIN_API_PROFILES"

_BUILTIN_PROFILES = {
    "QwenCloud (Singapore)": {
        "provider": "QwenCloud",
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "auth": "bearer_env",
        "api_key_env": "DASHSCOPE_API_KEY",
        "thinking_mode": "qwen",
    },
    "OpenRouter": {
        "provider": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "auth": "bearer_env",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "Together AI": {
        "provider": "Together AI",
        "base_url": "https://api.together.ai/v1",
        "auth": "bearer_env",
        "api_key_env": "TOGETHER_API_KEY",
    },
    "Fireworks AI": {
        "provider": "Fireworks AI",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "auth": "bearer_env",
        "api_key_env": "FIREWORKS_API_KEY",
    },
    "DeepInfra": {
        "provider": "DeepInfra",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "auth": "bearer_env",
        "api_key_env": "DEEPINFRA_TOKEN",
    },
    "Groq": {
        "provider": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "auth": "bearer_env",
        "api_key_env": "GROQ_API_KEY",
    },
    "Featherless": {
        "provider": "Featherless",
        "base_url": "https://api.featherless.ai/v1",
        "auth": "bearer_env",
        "api_key_env": "FEATHERLESS_API_KEY",
    },
    "OrcaRouter": {
        "provider": "OrcaRouter",
        "base_url": "https://api.orcarouter.ai/v1",
        "auth": "bearer_env",
        "api_key_env": "ORCAROUTER_API_KEY",
    },
    "Ollama (local)": {
        "provider": "Ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "auth": "none",
    },
    "vLLM (local)": {
        "provider": "vLLM",
        "base_url": "http://127.0.0.1:8000/v1",
        "auth": "none",
    },
    "llama.cpp (local)": {
        "provider": "llama.cpp",
        "base_url": "http://127.0.0.1:8080/v1",
        "auth": "none",
    },
}


def _profile_file_path() -> Path:
    configured = os.environ.get(PROFILE_FILE_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_PROFILE_FILE


def _validate_base_url(base_url: str, auth: str) -> str:
    value = str(base_url or "").strip().rstrip("/")
    if not value:
        raise ValueError("API profile base_url must not be empty.")

    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid API profile base_url: {value!r}")
    if parsed.username or parsed.password:
        raise ValueError("API profile base_url must not contain embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("API profile base_url must not contain a query string or fragment.")
    if auth != "none" and parsed.scheme != "https":
        raise ValueError("Authenticated API profiles must use HTTPS.")
    return value


def _normalize_profile(name: str, raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"API profile {name!r} must be a JSON object.")

    forbidden_secret_fields = {
        "api_key",
        "token",
        "bearer_token",
        "authorization",
        "headers",
        "secret",
    }
    forbidden_present = sorted(forbidden_secret_fields.intersection(raw))
    if forbidden_present:
        raise ValueError(
            f"API profile {name!r} contains forbidden secret/header fields: "
            + ", ".join(forbidden_present)
            + ". Store credentials only in server environment variables and use api_key_env."
        )

    provider = str(raw.get("provider") or name).strip()
    auth = str(raw.get("auth") or "bearer_env").strip().lower()
    if auth not in {"bearer_env", "none"}:
        raise ValueError(
            f"API profile {name!r} has unsupported auth mode {auth!r}; "
            "use 'bearer_env' or 'none'."
        )

    base_url = _validate_base_url(raw.get("base_url", ""), auth)
    api_key_env = str(raw.get("api_key_env") or "").strip()
    if auth == "bearer_env" and not api_key_env:
        raise ValueError(f"API profile {name!r} requires api_key_env for bearer_env auth.")
    if auth == "none":
        api_key_env = ""

    allowed_models = raw.get("allowed_models")
    if allowed_models is not None:
        if not isinstance(allowed_models, list) or not all(
            isinstance(item, str) and item.strip() for item in allowed_models
        ):
            raise ValueError(
                f"API profile {name!r} allowed_models must be a list of non-empty strings."
            )
        allowed_models = [item.strip() for item in allowed_models]

    max_tokens_limit = raw.get("max_tokens_limit")
    if max_tokens_limit is not None:
        if isinstance(max_tokens_limit, bool) or not isinstance(max_tokens_limit, int):
            raise ValueError(f"API profile {name!r} max_tokens_limit must be an integer.")
        if max_tokens_limit < 1:
            raise ValueError(f"API profile {name!r} max_tokens_limit must be >= 1.")

    allow_extra_body = bool(raw.get("allow_extra_body", False))
    thinking_mode = str(raw.get("thinking_mode") or "").strip().lower()
    send_seed = bool(raw.get("send_seed", False))

    return {
        "name": name,
        "provider": provider,
        "base_url": base_url,
        "auth": auth,
        "api_key_env": api_key_env,
        "allowed_models": allowed_models,
        "max_tokens_limit": max_tokens_limit,
        "allow_extra_body": allow_extra_body,
        "thinking_mode": thinking_mode,
        "send_seed": send_seed,
    }


def _load_api_profiles() -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    if os.environ.get(DISABLE_BUILTINS_ENV, "").strip() != "1":
        for name, raw in _BUILTIN_PROFILES.items():
            profiles[name] = _normalize_profile(name, raw)

    path = _profile_file_path()
    if not path.exists():
        return profiles

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ThinkingLLM API] Failed to load API profiles from {path}: {exc}")
        return profiles

    custom_profiles = payload.get("profiles", payload) if isinstance(payload, dict) else {}
    if not isinstance(custom_profiles, dict):
        print(f"[ThinkingLLM API] Ignoring invalid API profile file {path}: expected object.")
        return profiles

    for name, raw in custom_profiles.items():
        profile_name = str(name or "").strip()
        if not profile_name:
            continue
        try:
            profiles[profile_name] = _normalize_profile(profile_name, raw)
        except ValueError as exc:
            print(f"[ThinkingLLM API] Ignoring invalid profile {profile_name!r}: {exc}")

    return profiles


def _resolve_profile(profile_name: str) -> dict:
    name = str(profile_name or "").strip()
    profiles = _load_api_profiles()
    profile = profiles.get(name)
    if profile is None:
        print(
            f"[ThinkingLLM API] Unknown API profile {name!r}; "
            f"server profile file is {_profile_file_path()}."
        )
        raise ValueError("Unknown or unavailable server-side API profile.")
    return profile


def _authorization_headers(profile: dict) -> dict[str, str]:
    auth = profile["auth"]
    if auth == "none":
        return {}

    env_name = profile["api_key_env"]
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        print(
            f"[ThinkingLLM API] Profile {profile['name']!r} requires "
            f"server environment variable {env_name!r}, but it is not set."
        )
        raise RuntimeError("API credential is not configured for the selected server-side profile.")
    return {"Authorization": f"Bearer {api_key}"}


def _parse_extra_json(extra_body_json: str, *, allowed: bool) -> dict:
    raw = str(extra_body_json or "").strip()
    if not raw:
        return {}
    if not allowed:
        raise ValueError("extra_body_json is disabled by the selected server-side API profile.")

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"extra_body_json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("extra_body_json must contain a JSON object.")

    reserved = sorted(_RESERVED_EXTRA_BODY_FIELDS.intersection(value))
    if reserved:
        raise ValueError(
            "extra_body_json cannot override protected request fields: "
            + ", ".join(reserved)
        )
    return value


def _validate_model(profile: dict, model_name: str) -> str:
    model = str(model_name or "").strip()
    if not model:
        raise ValueError("model_name must not be empty.")

    allowed_models = profile.get("allowed_models")
    if allowed_models is not None and model not in allowed_models:
        raise ValueError(
            f"Model {model!r} is not allowed by API profile {profile['name']!r}."
        )
    return model


def _validate_max_tokens(profile: dict, max_tokens: int) -> int:
    requested = int(max_tokens)
    limit = profile.get("max_tokens_limit")
    if limit is not None and requested > limit:
        raise ValueError(
            f"Requested max_tokens={requested} exceeds server-side limit "
            f"{limit} for API profile {profile['name']!r}."
        )
    return requested


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
    """Call a server-approved OpenAI-compatible chat-completions profile."""

    @classmethod
    def INPUT_TYPES(cls):
        profiles = list(_load_api_profiles())
        if not profiles:
            profiles = ["No API profiles configured"]
        default_profile = (
            "QwenCloud (Singapore)"
            if "QwenCloud (Singapore)" in profiles
            else profiles[0]
        )
        return {
            "required": {
                "api_profile": (
                    profiles,
                    {
                        "default": default_profile,
                        "tooltip": (
                            "Server-side API profile. The profile binds provider endpoint, "
                            "authentication mode, and API-key environment variable. Secrets "
                            "and arbitrary endpoints are never accepted from workflow JSON."
                        ),
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "qwen3.8-max-preview",
                        "multiline": False,
                        "tooltip": (
                            "Exact model ID exposed by the selected provider. A server-side "
                            "profile may restrict this to an allowlist."
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
                        "tooltip": (
                            "Maximum output tokens requested. A server-side profile may "
                            "enforce a lower hard limit."
                        ),
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
                        "tooltip": (
                            "Used only when the selected server-side profile explicitly enables "
                            "send_seed."
                        ),
                    },
                ),
                "enable_thinking": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "For QwenCloud profiles this sends enable_thinking. Other providers "
                            "can use profile-approved extra_body_json when enabled by the admin."
                        ),
                    },
                ),
                "thinking_budget": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 0,
                        "max": 262144,
                        "tooltip": (
                            "QwenCloud thinking budget. Ignored for profiles that do not use "
                            "Qwen thinking parameters."
                        ),
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
                "extra_body_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Optional provider-specific JSON. Protected request fields cannot "
                            "be overridden, and server-side profiles can disable this entirely."
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
        "Calls a server-approved OpenAI-compatible /chat/completions profile. "
        "API keys and remote endpoint configuration stay server-side."
    )

    def generate(
        self,
        api_profile,
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
        extra_body_json="",
    ):
        profile = _resolve_profile(api_profile)
        model = _validate_model(profile, model_name)
        requested_max_tokens = _validate_max_tokens(profile, max_tokens)
        endpoint = profile["base_url"] + "/chat/completions"
        auth_headers = _authorization_headers(profile)

        messages = []
        if str(system_prompt or "").strip():
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(prompt or "")})

        request_body = {
            "model": model,
            "messages": messages,
            "max_tokens": requested_max_tokens,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": True,
        }
        if profile.get("send_seed"):
            request_body["seed"] = int(seed)

        if profile.get("thinking_mode") == "qwen":
            request_body["enable_thinking"] = bool(enable_thinking)
            if int(thinking_budget) > 0:
                request_body["thinking_budget"] = int(thinking_budget)

        extra = _parse_extra_json(
            extra_body_json,
            allowed=bool(profile.get("allow_extra_body", False)),
        )
        request_body.update(extra)
        request_body["stream"] = True

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "ComfyUI-ThinkingLLM/OpenAI-Compatible-API",
        }
        headers.update(auth_headers)

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(request_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        display = get_thinking_stream_display() if stream_tokens_to_terminal else None
        if display is not None:
            display.start_stage(f"API {profile['provider']} / {model}")

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
            body = _read_error_body(exc)[:4096]
            detail = f"HTTP {exc.code} {exc.reason}"
            if body:
                detail += f": {body}"
            print(
                f"[ThinkingLLM API] Request failed for profile {profile['name']!r}: {detail}"
            )
            raise RuntimeError(
                f"ThinkingLLM API request failed for profile {profile['name']!r} "
                f"with HTTP status {exc.code}."
            ) from exc
        except urllib.error.URLError as exc:
            print(
                f"[ThinkingLLM API] Network error for profile {profile['name']!r}: {exc.reason}"
            )
            raise RuntimeError(
                f"ThinkingLLM API network request failed for profile {profile['name']!r}."
            ) from exc
        finally:
            if display is not None:
                display.end_stage()

        content_text = "".join(content_parts).strip()
        reasoning_text = "".join(reasoning_parts).strip()
        raw_trace = compose_streamed_model_output(reasoning_text, content_text)

        # Some compatible backends emit literal <think> blocks inside content
        # rather than exposing a separate reasoning channel. Keep RAW_TRACE
        # untouched while stripping the think block from the normal RESPONSE.
        response_text = clean_model_output(
            content_text,
            OutputCleanConfig(
                mode="text",
                strip_think=True,
                strip_code_fences=False,
                strip_role_prefixes=False,
                strip_json_wrappers=False,
                strip_leading_preamble=False,
                strip_planning=False,
            ),
        )

        return (response_text, raw_trace)


NODE_CLASS_MAPPINGS = {
    "ThinkingLLM_OpenAICompatibleAPI": ThinkingLLMOpenAICompatibleAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThinkingLLM_OpenAICompatibleAPI": "ThinkingLLM API (OpenAI Compatible)",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
