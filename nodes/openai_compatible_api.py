"""Secure OpenAI-compatible API node for ThinkingLLM.

Workflow/API input never receives provider credentials, credential environment
variable names, or arbitrary provider endpoints. It selects a server-side
profile that binds endpoint, authentication, capabilities, and request limits.
"""
from __future__ import annotations

import base64
import io
import ipaddress
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from AILab_OutputCleaner import OutputCleanConfig, clean_model_output
from AILab_StreamDisplay import compose_streamed_model_output, extract_stream_token, get_thinking_stream_display

PACKAGE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE_FILE = PACKAGE_DIR / "thinkingllm_api_profiles.json"
PROFILE_FILE_ENV = "THINKINGLLM_API_PROFILES_FILE"
DISABLE_BUILTINS_ENV = "THINKINGLLM_DISABLE_BUILTIN_API_PROFILES"
DEFAULT_MAX_INPUT_CHARS = 1_000_000
DEFAULT_MAX_TIMEOUT_SECONDS = 300
DEFAULT_MAX_IMAGE_PIXELS = 16_777_216
DEFAULT_MAX_IMAGE_BYTES = 20_000_000
MAX_EXTRA_BODY_JSON_CHARS = 65_536
MAX_MODEL_NAME_CHARS = 512
MAX_PROFILE_LABEL_CHARS = 128
MAX_IMAGE_URL_CHARS = 8_192

# Curated convenience model catalog for the node UI (see web/api_model_catalogs.json).
# It is UX only: it never overrides a server-side profile's allowed_models allowlist.
CATALOG_PATH = PACKAGE_DIR / "web" / "api_model_catalogs.json"
_CATALOG_CACHE: dict | None = None

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_IMAGE_ERROR_RE = re.compile(
    r"(?i)(image|vision|multimodal|image_url|modality|unsupported\s+(?:content|input)|invalid\s+(?:content|input))"
)
_RESERVED_EXTRA_BODY_FIELDS = {
    "model", "messages", "stream", "max_tokens", "max_completion_tokens",
    "temperature", "top_p", "seed", "enable_thinking", "thinking_budget",
}
_PROFILE_ALLOWED_FIELDS = {
    "provider", "base_url", "auth", "api_key_env", "allowed_models",
    "max_tokens_limit", "max_input_chars", "max_timeout_seconds", "max_thinking_budget",
    "allowed_extra_body_fields", "send_seed", "thinking_mode",
    "max_tokens_field", "allow_insecure_http",
    "allow_images", "allow_image_url", "max_image_pixels", "max_image_bytes",
}


def _cloud_profile(provider: str, base_url: str, api_key_env: str, **extra) -> dict:
    profile = {
        "provider": provider,
        "base_url": base_url,
        "auth": "bearer_env",
        "api_key_env": api_key_env,
        "allow_images": True,
        "allow_image_url": True,
    }
    profile.update(extra)
    return profile


def _local_profile(provider: str, base_url: str, *, send_seed: bool = False) -> dict:
    return {
        "provider": provider,
        "base_url": base_url,
        "auth": "none",
        "allow_images": True,
        "allow_image_url": True,
        "send_seed": send_seed,
    }


_BUILTIN_PROFILES = {
    "OpenAI": _cloud_profile(
        "OpenAI", "https://api.openai.com/v1", "OPENAI_API_KEY",
        max_tokens_field="max_completion_tokens",
    ),
    "QwenCloud (Singapore)": _cloud_profile(
        "QwenCloud", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY", thinking_mode="qwen",
    ),
    "OpenRouter": _cloud_profile(
        "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
    ),
    "Together AI": _cloud_profile(
        "Together AI", "https://api.together.ai/v1", "TOGETHER_API_KEY"
    ),
    "Fireworks AI": _cloud_profile(
        "Fireworks AI", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"
    ),
    "DeepInfra": _cloud_profile(
        "DeepInfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_TOKEN"
    ),
    "Groq": _cloud_profile(
        "Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"
    ),
    "Featherless": _cloud_profile(
        "Featherless", "https://api.featherless.ai/v1", "FEATHERLESS_API_KEY"
    ),
    "OrcaRouter": _cloud_profile(
        "OrcaRouter", "https://api.orcarouter.ai/v1", "ORCAROUTER_API_KEY"
    ),
    "Ollama (local)": _local_profile("Ollama", "http://127.0.0.1:11434/v1"),
    "vLLM (local)": _local_profile("vLLM", "http://127.0.0.1:8000/v1", send_seed=True),
    "llama.cpp (local)": _local_profile("llama.cpp", "http://127.0.0.1:8080/v1", send_seed=True),
}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects so bearer credentials never cross an origin boundary."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler())


class _ProviderStreamError(RuntimeError):
    pass


def _profile_file_path() -> Path:
    configured = os.environ.get(PROFILE_FILE_ENV, "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_PROFILE_FILE


def _has_control_chars(value: str) -> bool:
    return bool(_CONTROL_CHAR_RE.search(str(value or "")))


def _safe_label(value: str, field: str, max_chars: int = MAX_PROFILE_LABEL_CHARS) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must not be empty.")
    if len(text) > max_chars:
        raise ValueError(f"{field} is too long (maximum {max_chars} characters).")
    if _has_control_chars(text):
        raise ValueError(f"{field} must not contain control characters.")
    return text


def _strict_bool(raw: dict, key: str, default: bool = False) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"API profile {key} must be a JSON boolean.")
    return value


def _profile_int(raw: dict, key: str, default=None, minimum: int = 1, maximum: int | None = None):
    value = raw.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"API profile {key} must be an integer.")
    if value < minimum or (maximum is not None and value > maximum):
        range_text = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ValueError(f"API profile {key} must be {range_text}.")
    return value


def _is_loopback_host(hostname: str | None) -> bool:
    host = str(hostname or "").strip().rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_base_url(base_url: str, auth: str, allow_insecure_http: bool = False) -> str:
    value = str(base_url or "").strip().rstrip("/")
    if not value or _has_control_chars(value):
        raise ValueError("API profile base_url is empty or contains control characters.")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ValueError(f"Invalid API profile base_url: {value!r}")
    if parsed.username or parsed.password:
        raise ValueError("API profile base_url must not contain embedded credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("API profile base_url must not contain a query string or fragment.")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("API profile base_url contains an invalid port.") from exc
    if auth != "none" and parsed.scheme != "https":
        raise ValueError("Authenticated API profiles must use HTTPS.")
    if parsed.scheme == "http" and auth == "none" and not _is_loopback_host(parsed.hostname) and not allow_insecure_http:
        raise ValueError(
            "Plain HTTP is allowed only for loopback endpoints by default. "
            "Set allow_insecure_http=true only for administrator-approved cleartext LAN/internal traffic."
        )
    return value


def _validate_model_identifier(value: str, field: str = "model_name") -> str:
    model = str(value or "").strip()
    if not model:
        raise ValueError(f"{field} must not be empty.")
    if len(model) > MAX_MODEL_NAME_CHARS:
        raise ValueError(f"{field} is too long (maximum {MAX_MODEL_NAME_CHARS} characters).")
    if _has_control_chars(model):
        raise ValueError(f"{field} must not contain control characters.")
    return model


def _normalize_profile(name: str, raw: dict) -> dict:
    profile_name = _safe_label(name, "API profile name")
    if not isinstance(raw, dict):
        raise ValueError(f"API profile {profile_name!r} must be a JSON object.")
    forbidden = sorted({"api_key", "token", "bearer_token", "authorization", "headers", "secret"} & set(raw))
    if forbidden:
        raise ValueError(f"API profile {profile_name!r} contains forbidden secret/header fields: {', '.join(forbidden)}.")
    if "allow_extra_body" in raw:
        raise ValueError(f"API profile {profile_name!r} uses deprecated allow_extra_body; use allowed_extra_body_fields.")
    unknown = sorted(set(raw) - _PROFILE_ALLOWED_FIELDS)
    if unknown:
        raise ValueError(f"API profile {profile_name!r} contains unknown fields: {', '.join(unknown)}")

    provider = _safe_label(raw.get("provider") or profile_name, "provider")
    auth = str(raw.get("auth") or "bearer_env").strip().lower()
    if auth not in {"bearer_env", "none"}:
        raise ValueError(f"API profile {profile_name!r} auth must be 'bearer_env' or 'none'.")
    allow_insecure_http = _strict_bool(raw, "allow_insecure_http", False)
    base_url = _validate_base_url(raw.get("base_url", ""), auth, allow_insecure_http)

    api_key_env = str(raw.get("api_key_env") or "").strip()
    if auth == "bearer_env":
        if not _ENV_NAME_RE.fullmatch(api_key_env):
            raise ValueError(f"API profile {profile_name!r} requires a valid environment-variable name in api_key_env.")
    elif api_key_env:
        raise ValueError(f"API profile {profile_name!r} must not set api_key_env when auth='none'.")

    raw_models = raw.get("allowed_models")
    allowed_models = None
    if raw_models is not None:
        if not isinstance(raw_models, list) or not all(isinstance(item, str) for item in raw_models):
            raise ValueError(f"API profile {profile_name!r} allowed_models must be a list of strings.")
        allowed_models = [_validate_model_identifier(item, "allowed model") for item in raw_models]

    raw_extra = raw.get("allowed_extra_body_fields") or []
    if not isinstance(raw_extra, list) or not all(isinstance(item, str) for item in raw_extra):
        raise ValueError(f"API profile {profile_name!r} allowed_extra_body_fields must be a list of strings.")
    allowed_extra = []
    for item in raw_extra:
        field = item.strip()
        if not field or len(field) > 128 or _has_control_chars(field):
            raise ValueError(f"API profile {profile_name!r} contains an invalid extra-body field name.")
        if field in _RESERVED_EXTRA_BODY_FIELDS:
            raise ValueError(f"API profile {profile_name!r} cannot allow protected extra-body field {field!r}.")
        if field not in allowed_extra:
            allowed_extra.append(field)

    thinking_mode = str(raw.get("thinking_mode") or "").strip().lower()
    if thinking_mode not in {"", "qwen"}:
        raise ValueError(f"API profile {profile_name!r} has unsupported thinking_mode {thinking_mode!r}.")
    max_tokens_field = str(raw.get("max_tokens_field") or "max_tokens").strip()
    if max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError(f"API profile {profile_name!r} has invalid max_tokens_field.")

    allow_images = _strict_bool(raw, "allow_images", False)
    allow_image_url = _strict_bool(raw, "allow_image_url", False)
    if allow_image_url and not allow_images:
        raise ValueError(f"API profile {profile_name!r} cannot enable allow_image_url while allow_images is false.")

    return {
        "name": profile_name,
        "provider": provider,
        "base_url": base_url,
        "auth": auth,
        "api_key_env": api_key_env,
        "allowed_models": allowed_models,
        "max_tokens_limit": _profile_int(raw, "max_tokens_limit", None, maximum=65_536),
        "max_input_chars": _profile_int(raw, "max_input_chars", DEFAULT_MAX_INPUT_CHARS, maximum=20_000_000),
        "max_timeout_seconds": _profile_int(raw, "max_timeout_seconds", DEFAULT_MAX_TIMEOUT_SECONDS, maximum=3_600),
        "max_thinking_budget": _profile_int(raw, "max_thinking_budget", 262_144, minimum=0, maximum=262_144),
        "allowed_extra_body_fields": allowed_extra,
        "thinking_mode": thinking_mode,
        "send_seed": _strict_bool(raw, "send_seed", False),
        "max_tokens_field": max_tokens_field,
        "allow_insecure_http": allow_insecure_http,
        "allow_images": allow_images,
        "allow_image_url": allow_image_url,
        "max_image_pixels": _profile_int(raw, "max_image_pixels", DEFAULT_MAX_IMAGE_PIXELS, maximum=100_000_000),
        "max_image_bytes": _profile_int(raw, "max_image_bytes", DEFAULT_MAX_IMAGE_BYTES, maximum=100_000_000),
    }


def _load_api_profiles() -> dict[str, dict]:
    profiles = {}
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
    custom = payload.get("profiles", payload) if isinstance(payload, dict) else {}
    if not isinstance(custom, dict):
        print(f"[ThinkingLLM API] Ignoring invalid API profile file {path}: expected object.")
        return profiles
    for name, raw in custom.items():
        profile_name = str(name or "").strip()
        if not profile_name:
            continue
        profiles.pop(profile_name, None)
        try:
            profiles[profile_name] = _normalize_profile(profile_name, raw)
        except ValueError as exc:
            print(f"[ThinkingLLM API] Ignoring invalid profile {profile_name!r}: {exc}")
    return profiles


def _resolve_profile(profile_name: str) -> dict:
    name = str(profile_name or "").strip()
    profile = _load_api_profiles().get(name)
    if profile is None:
        print(f"[ThinkingLLM API] Unknown API profile {name!r}; profile file is {_profile_file_path()}.")
        raise ValueError("Unknown or unavailable server-side API profile.")
    return profile


def _load_model_catalog() -> dict:
    """Return the curated convenience model catalog (UX only, never an allowlist).

    Read-only and failure-tolerant: a missing or broken catalog must never
    block the node. The catalog only feeds the front-end dropdown; request
    security is enforced by the profile's allowed_models, not by this file.
    """
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    try:
        with CATALOG_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ThinkingLLM API] Failed to load model catalog from {CATALOG_PATH}: {exc}")
        _CATALOG_CACHE = {}
        return _CATALOG_CACHE
    profiles = payload.get("profiles") if isinstance(payload, dict) else None
    _CATALOG_CACHE = profiles if isinstance(profiles, dict) else {}
    return _CATALOG_CACHE


def _catalog_models_for_profile(profile_name: str) -> list[str]:
    """Convenience model IDs for a profile name, used by the node UI only."""
    entry = _load_model_catalog().get(str(profile_name or ""))
    if not isinstance(entry, dict):
        return []
    models = entry.get("models")
    if not isinstance(models, list):
        return []
    ids: list[str] = []
    for item in models:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return ids


def _catalog_all_model_ids() -> list[str]:
    """Union of every curated model ID across all profiles (deduplicated).

    Used as the backend combo list so that any saved model_name from any
    profile still passes ComfyUI's front-end validation ("Value not in list").
    The front-end narrows the visible dropdown to the selected profile.
    """
    seen: dict[str, None] = {}
    for entry in _load_model_catalog().values():
        if not isinstance(entry, dict):
            continue
        models = entry.get("models")
        if not isinstance(models, list):
            continue
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                seen.setdefault(item["id"], None)
    return list(seen)


def _credential_value(profile: dict) -> str:
    if profile["auth"] == "none":
        return ""
    env_name = profile["api_key_env"]
    value = os.environ.get(env_name, "").strip()
    if not value:
        print(f"[ThinkingLLM API] Profile {profile['name']!r} requires environment variable {env_name!r}, but it is not set.")
        raise RuntimeError("API credential is not configured for the selected server-side profile.")
    return value


def _authorization_headers(profile: dict) -> dict[str, str]:
    value = _credential_value(profile)
    return {"Authorization": f"Bearer {value}"} if value else {}


def _reject_json_constant(value: str):
    raise ValueError(f"Non-finite JSON constant {value!r} is not allowed.")


def _parse_extra_json(extra_body_json: str, allowed_fields) -> dict:
    raw = str(extra_body_json or "").strip()
    if not raw:
        return {}
    if len(raw) > MAX_EXTRA_BODY_JSON_CHARS:
        raise ValueError(f"extra_body_json exceeds the {MAX_EXTRA_BODY_JSON_CHARS}-character safety limit.")
    allowed = set(allowed_fields or [])
    if not allowed:
        raise ValueError("extra_body_json is disabled by the selected server-side API profile.")
    try:
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"extra_body_json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("extra_body_json must contain a JSON object.")
    reserved = sorted(_RESERVED_EXTRA_BODY_FIELDS & set(value))
    if reserved:
        raise ValueError(f"extra_body_json cannot override protected request fields: {', '.join(reserved)}")
    unapproved = sorted(set(value) - allowed)
    if unapproved:
        raise ValueError(f"extra_body_json contains fields not allowed by the server-side profile: {', '.join(unapproved)}")
    return value


def _validate_model(profile: dict, model_name: str) -> str:
    model = _validate_model_identifier(model_name)
    if profile.get("allowed_models") is not None and model not in profile["allowed_models"]:
        raise ValueError(f"Model {model!r} is not allowed by API profile {profile['name']!r}.")
    return model


def _validate_max_tokens(profile: dict, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65_536:
        raise ValueError("max_tokens must be an integer between 1 and 65536.")
    limit = profile.get("max_tokens_limit")
    if limit is not None and value > limit:
        raise ValueError(f"Requested max_tokens={value} exceeds server-side limit {limit} for API profile {profile['name']!r}.")
    return value


def _validate_timeout(profile: dict, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 3_600:
        raise ValueError("timeout_seconds must be an integer between 1 and 3600.")
    limit = profile.get("max_timeout_seconds", DEFAULT_MAX_TIMEOUT_SECONDS)
    if value > limit:
        raise ValueError(f"Requested timeout_seconds={value} exceeds server-side limit {limit} for API profile {profile['name']!r}.")
    return value


def _validate_sampling(temperature, top_p, seed, thinking_budget):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a numeric value.")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise ValueError("top_p must be a numeric value.")
    t, p = float(temperature), float(top_p)
    if not math.isfinite(t) or not 0.0 <= t <= 2.0:
        raise ValueError("temperature must be a finite value between 0.0 and 2.0.")
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("top_p must be a finite value between 0.0 and 1.0.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in the supported range.")
    if isinstance(thinking_budget, bool) or not isinstance(thinking_budget, int) or not 0 <= thinking_budget <= 262_144:
        raise ValueError("thinking_budget must be an integer between 0 and 262144.")
    return t, p, seed, thinking_budget


def _validate_thinking_budget(profile: dict, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 262_144:
        raise ValueError("thinking_budget must be an integer between 0 and 262144.")
    limit = profile.get("max_thinking_budget", 262_144)
    if value > limit:
        raise ValueError(
            f"Requested thinking_budget={value} exceeds server-side limit {limit} "
            f"for API profile {profile['name']!r}."
        )
    return value


def _validate_input_text(profile: dict, system_prompt, prompt) -> tuple[str, str]:
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string.")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("prompt must be a string.")
    system_text, prompt_text = system_prompt or "", prompt or ""
    total = len(system_text) + len(prompt_text)
    limit = profile.get("max_input_chars", DEFAULT_MAX_INPUT_CHARS)
    if total > limit:
        raise ValueError(f"Combined system_prompt + prompt length ({total}) exceeds server-side max_input_chars={limit} for API profile {profile['name']!r}.")
    return system_text, prompt_text


def _validate_image_url(profile: dict, image_url) -> str:
    value = str(image_url or "").strip()
    if not value:
        return ""
    if not profile.get("allow_images", False):
        raise ValueError(f"Image input is disabled by API profile {profile['name']!r}.")
    if not profile.get("allow_image_url", False):
        raise ValueError(f"image_url input is disabled by API profile {profile['name']!r}.")
    if len(value) > MAX_IMAGE_URL_CHARS or _has_control_chars(value):
        raise ValueError("image_url is too long or contains control characters.")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or not parsed.hostname:
        raise ValueError("image_url must be a public HTTPS URL.")
    if parsed.username or parsed.password:
        raise ValueError("image_url must not contain embedded credentials.")
    if parsed.fragment:
        raise ValueError("image_url must not contain a URL fragment.")
    return value


def _image_to_data_url(profile: dict, image) -> str:
    if image is None:
        return ""
    if not profile.get("allow_images", False):
        raise ValueError(f"Image input is disabled by API profile {profile['name']!r}.")

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Image input requires NumPy and Pillow in the ComfyUI Python environment.") from exc

    shape = tuple(int(part) for part in getattr(image, "shape", ()))
    if len(shape) == 4:
        if shape[0] != 1:
            raise ValueError("The remote API node currently accepts exactly one IMAGE, not an image batch.")
        height, width, channels = shape[1], shape[2], shape[3]
        frame = image[0]
    elif len(shape) == 3:
        height, width, channels = shape
        frame = image
    else:
        raise ValueError("IMAGE input must have shape [B,H,W,C] or [H,W,C].")

    if height < 1 or width < 1 or channels not in {1, 3, 4}:
        raise ValueError("IMAGE input has unsupported dimensions or channel count.")
    pixels = height * width
    pixel_limit = profile.get("max_image_pixels", DEFAULT_MAX_IMAGE_PIXELS)
    if pixels > pixel_limit:
        raise ValueError(
            f"IMAGE input has {pixels} pixels, exceeding server-side max_image_pixels={pixel_limit} "
            f"for API profile {profile['name']!r}."
        )

    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    if hasattr(frame, "numpy"):
        frame = frame.numpy()
    array = np.asarray(frame, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] not in {1, 3, 4}:
        raise ValueError("IMAGE input could not be converted to a supported HWC array.")
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)

    if array.shape[2] == 1:
        array = array[:, :, 0]
        mode = "L"
    elif array.shape[2] == 4:
        mode = "RGBA"
    else:
        mode = "RGB"

    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    encoded_bytes = buffer.getvalue()
    byte_limit = profile.get("max_image_bytes", DEFAULT_MAX_IMAGE_BYTES)
    if len(encoded_bytes) > byte_limit:
        raise ValueError(
            f"PNG-encoded IMAGE is {len(encoded_bytes)} bytes, exceeding server-side "
            f"max_image_bytes={byte_limit} for API profile {profile['name']!r}."
        )
    return "data:image/png;base64," + base64.b64encode(encoded_bytes).decode("ascii")


def _build_user_content(profile: dict, prompt_text: str, image=None, image_url=""):
    url = str(image_url or "").strip()
    if image is not None and url:
        raise ValueError("Use either the IMAGE input or image_url, not both in the same request.")
    if image is None and not url:
        return prompt_text, False

    if image is not None:
        source = _image_to_data_url(profile, image)
    else:
        source = _validate_image_url(profile, url)

    return [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": source}},
    ], True


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        data = exc.read()
    except Exception:
        return ""
    return data.decode("utf-8", errors="replace").strip() if isinstance(data, bytes) else str(data or "").strip()


def _terminal_display_safe(text: str) -> str:
    return "".join(ch for ch in str(text or "") if ch in {"\n", "\t"} or (ord(ch) >= 0x20 and not 0x7F <= ord(ch) <= 0x9F))


def _redact_for_log(text: str, profile: dict) -> str:
    value = _terminal_display_safe(text)
    if profile.get("auth") == "bearer_env":
        secret = os.environ.get(profile.get("api_key_env", ""), "")
        if secret:
            value = value.replace(secret, "<redacted>")
    return value[:4096]


def _reasoning_details_text(container) -> str:
    if not isinstance(container, dict) or not isinstance(container.get("reasoning_details"), list):
        return ""
    parts = []
    for item in container["reasoning_details"]:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            for key in ("text", "summary", "content", "reasoning"):
                if isinstance(item.get(key), str) and item[key]:
                    parts.append(item[key])
                    break
    return "".join(parts)


def _extract_stream_token_extended(chunk) -> dict[str, str]:
    token = extract_stream_token(chunk)
    if token.get("reasoning") or not isinstance(chunk, dict):
        return token
    choices = chunk.get("choices") or []
    choice = choices[0] if isinstance(choices, list) and choices else {}
    delta = choice.get("delta") if isinstance(choice, dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    for container in (delta, message, choice, chunk):
        reasoning = _reasoning_details_text(container)
        if reasoning:
            return {"reasoning": reasoning, "content": token.get("content", "")}
    return token


def _provider_error_from_payload(payload) -> str:
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if error:
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message") or error.get("type") or "provider stream error"
            return f"{code}: {message}" if code is not None else str(message)
        return str(error)
    choices = payload.get("choices") or []
    if isinstance(choices, list) and choices and isinstance(choices[0], dict) and choices[0].get("finish_reason") == "error":
        return "provider reported finish_reason=error"
    return ""


def _extract_nonstream_response(payload: dict) -> tuple[str, str]:
    error = _provider_error_from_payload(payload)
    if error:
        raise _ProviderStreamError(error)
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content") or choice.get("text") or ""
    reasoning = (
        message.get("reasoning_content") or message.get("reasoning") or message.get("thinking")
        or choice.get("reasoning_content") or _reasoning_details_text(message)
        or _reasoning_details_text(choice) or ""
    )
    return str(content or ""), str(reasoning or "")


def _open_request(req: urllib.request.Request, timeout_seconds: int):
    return _NO_REDIRECT_OPENER.open(req, timeout=timeout_seconds)


def _image_rejection_message(profile: dict, model: str, status=None) -> str:
    status_text = f" (HTTP {status})" if status is not None else ""
    return (
        f"Image input was rejected by API profile {profile['name']!r} / model {model!r}{status_text}. "
        "Use a vision-capable model supported by that provider, or remove the IMAGE/image_url input."
    )


def _looks_like_image_error(detail: str) -> bool:
    return bool(_IMAGE_ERROR_RE.search(str(detail or "")))


class ThinkingLLMOpenAICompatibleAPI:
    @classmethod
    def INPUT_TYPES(cls):
        profiles = list(_load_api_profiles()) or ["No API profiles configured"]
        default_profile = "OpenRouter" if "OpenRouter" in profiles else profiles[0]
        default_models = _catalog_models_for_profile(default_profile)
        if not default_models:
            default_models = ["provider/model-id"]
        # Use the union of all profile models so any saved model_name from any
        # profile passes ComfyUI's front-end combo validation. The JS narrows
        # the visible dropdown to the selected api_profile.
        all_models = _catalog_all_model_ids()
        if not all_models:
            all_models = default_models
        return {
            "required": {
                "api_profile": (profiles, {"default": default_profile, "tooltip": "Server-side profile binding endpoint, authentication, credential reference, capabilities, and safety limits."}),
                "model_name": (all_models, {"default": default_models[0], "tooltip": "Curated model for the selected profile. For anything not listed, leave this as-is and type the ID in custom_model_name below."}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 65536}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 2**32 - 1, "tooltip": "Sent only when the server profile enables send_seed."}),
                "enable_thinking": ("BOOLEAN", {"default": True, "tooltip": "Used by profiles with thinking_mode='qwen'."}),
                "thinking_budget": ("INT", {"default": 8192, "min": 0, "max": 262144}),
                "stream_tokens_to_terminal": ("BOOLEAN", {"default": False, "tooltip": "Print a control-character-sanitized display copy to the ComfyUI terminal."}),
                "timeout_seconds": ("INT", {"default": 300, "min": 1, "max": 3600, "tooltip": "Server profile enforces max_timeout_seconds (300 by default)."}),
            },
            "optional": {
                "custom_model_name": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional: exact model ID for anything not in the curated list. When set, it overrides model_name."}),
                "image": ("IMAGE", {"tooltip": "Single ComfyUI IMAGE. Encoded as PNG Base64 data URL and sent as OpenAI-style image_url content."}),
                "image_url": ("STRING", {"default": "", "multiline": False, "tooltip": "Alternative to IMAGE: public HTTPS image URL. Do not connect IMAGE at the same time."}),
                "extra_body_json": ("STRING", {"default": "", "multiline": True, "tooltip": "Every top-level field must be explicitly allowlisted by the server profile."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "RAW_TRACE")
    FUNCTION = "generate"
    CATEGORY = "ThinkingLLM/API"
    DESCRIPTION = "Calls a server-approved OpenAI-compatible /chat/completions profile with optional image input; credentials and endpoint configuration stay server-side."

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def generate(
        self, api_profile, model_name, system_prompt, prompt, max_tokens, temperature, top_p,
        seed, enable_thinking, thinking_budget, stream_tokens_to_terminal, timeout_seconds,
        custom_model_name="", image=None, image_url="", extra_body_json="",
    ):
        profile = _resolve_profile(api_profile)
        if not isinstance(enable_thinking, bool) or not isinstance(stream_tokens_to_terminal, bool):
            raise ValueError("Boolean node inputs must be booleans.")
        # A custom model ID overrides the curated dropdown selection.
        if str(custom_model_name or "").strip():
            model_name = custom_model_name
        model = _validate_model(profile, model_name)
        max_out = _validate_max_tokens(profile, max_tokens)
        timeout = _validate_timeout(profile, timeout_seconds)
        temperature, top_p, seed, thinking_budget = _validate_sampling(temperature, top_p, seed, thinking_budget)
        thinking_budget = _validate_thinking_budget(profile, thinking_budget)
        system_text, prompt_text = _validate_input_text(profile, system_prompt, prompt)
        user_content, image_supplied = _build_user_content(profile, prompt_text, image=image, image_url=image_url)

        messages = []
        if system_text.strip():
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_content})
        request_body = {
            "model": model,
            "messages": messages,
            profile.get("max_tokens_field", "max_tokens"): max_out,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if profile.get("send_seed"):
            request_body["seed"] = seed
        if profile.get("thinking_mode") == "qwen":
            request_body["enable_thinking"] = enable_thinking
            if thinking_budget > 0:
                request_body["thinking_budget"] = thinking_budget
        request_body.update(_parse_extra_json(extra_body_json, profile.get("allowed_extra_body_fields", [])))
        request_body["stream"] = True

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "ComfyUI-ThinkingLLM/OpenAI-Compatible-API",
        }
        headers.update(_authorization_headers(profile))
        try:
            payload = json.dumps(request_body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("Unable to serialize the API request safely.") from exc
        req = urllib.request.Request(profile["base_url"] + "/chat/completions", data=payload, headers=headers, method="POST")

        reasoning_parts, content_parts = [], []
        display = get_thinking_stream_display() if stream_tokens_to_terminal else None
        if display is not None:
            display.start_stage(_terminal_display_safe(f"API {profile['provider']} / {model}"))
        deadline = time.monotonic() + timeout
        try:
            with _open_request(req, timeout) as response:
                if "text/event-stream" not in str(response.headers.get("Content-Type", "")).lower():
                    raw = response.read().decode("utf-8", errors="replace")
                    content, reasoning = _extract_nonstream_response(json.loads(raw, parse_constant=_reject_json_constant))
                    content_parts.append(content)
                    reasoning_parts.append(reasoning)
                    if display is not None:
                        display.push_compact(_terminal_display_safe(reasoning or content))
                else:
                    for raw_line in response:
                        if time.monotonic() > deadline:
                            raise TimeoutError("provider stream exceeded configured request deadline")
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or line.startswith(":"):
                            continue
                        line = line[5:].strip() if line.startswith("data:") else line
                        if line == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line, parse_constant=_reject_json_constant)
                        except json.JSONDecodeError:
                            continue
                        error = _provider_error_from_payload(chunk)
                        if error:
                            raise _ProviderStreamError(error)
                        token = _extract_stream_token_extended(chunk)
                        reasoning, content = token.get("reasoning", ""), token.get("content", "")
                        if reasoning:
                            reasoning_parts.append(reasoning)
                            if display is not None:
                                display.push_compact(_terminal_display_safe(reasoning))
                        if content:
                            content_parts.append(content)
                            if display is not None:
                                display.push_compact(_terminal_display_safe(content))
        except urllib.error.HTTPError as exc:
            detail = f"HTTP {exc.code} {exc.reason}"
            body = _redact_for_log(_read_error_body(exc), profile)
            if body:
                detail += f": {body}"
            print(f"[ThinkingLLM API] Request failed for profile {profile['name']!r}: {detail}")
            if 300 <= int(exc.code) < 400:
                raise RuntimeError(f"ThinkingLLM API redirect rejected for profile {profile['name']!r}.") from exc
            if image_supplied and (int(exc.code) in {400, 415, 422} or _looks_like_image_error(body)):
                raise RuntimeError(_image_rejection_message(profile, model, exc.code)) from exc
            raise RuntimeError(f"ThinkingLLM API request failed for profile {profile['name']!r} with HTTP status {exc.code}.") from exc
        except urllib.error.URLError as exc:
            print(f"[ThinkingLLM API] Network error for profile {profile['name']!r}: {_redact_for_log(str(exc.reason), profile)}")
            raise RuntimeError(f"ThinkingLLM API network request failed for profile {profile['name']!r}.") from exc
        except (TimeoutError, _ProviderStreamError) as exc:
            print(f"[ThinkingLLM API] Provider stream failed for profile {profile['name']!r}: {_redact_for_log(str(exc), profile)}")
            if image_supplied and _looks_like_image_error(str(exc)):
                raise RuntimeError(_image_rejection_message(profile, model)) from exc
            raise RuntimeError(f"ThinkingLLM API stream failed for profile {profile['name']!r}.") from exc
        finally:
            if display is not None:
                display.end_stage()

        content_text = "".join(content_parts).strip()
        reasoning_text = "".join(reasoning_parts).strip()
        raw_trace = compose_streamed_model_output(reasoning_text, content_text)
        response_text = clean_model_output(
            content_text,
            OutputCleanConfig(
                mode="text", strip_think=True, strip_code_fences=False, strip_role_prefixes=False,
                strip_json_wrappers=False, strip_leading_preamble=False, strip_planning=False,
            ),
        )
        return (response_text, raw_trace)


NODE_CLASS_MAPPINGS = {"ThinkingLLM_OpenAICompatibleAPI": ThinkingLLMOpenAICompatibleAPI}
NODE_DISPLAY_NAME_MAPPINGS = {"ThinkingLLM_OpenAICompatibleAPI": "ThinkingLLM API (OpenAI Compatible)"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
