"""Validate the ThinkingLLM API model catalogs.

Checks that web/api_model_catalogs.json stays well-formed, that every
cloud profile shipped by the API node has a curated entry, and that model
IDs are safe to send in a request body (no control characters, no
leading/trailing whitespace, no duplicates).

Run in CI on every PR, push to main, and before every release:

    python tools/check_api_model_catalogs.py

Exit code 0 on success, 1 on any validation failure.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "web" / "api_model_catalogs.json"

# Cloud profiles exposed by the API node (nodes/openai_compatible_api.py).
# Local endpoints (Ollama, vLLM, llama.cpp) are intentionally excluded
# because their model IDs are runtime/environment specific.
REQUIRED_PROFILE_KEYS = {
    "OpenAI": "OpenAI",
    "QwenCloud (Singapore)": "QwenCloud",
    "OpenRouter": "OpenRouter",
    "Together AI": "Together AI",
    "Fireworks AI": "Fireworks AI",
    "DeepInfra": "DeepInfra",
    "Groq": "Groq",
    "Featherless": "Featherless",
    "OrcaRouter": "OrcaRouter",
}

# Reasonable bound to catch accidental mass-paste or generated blobs.
# Public providers ship full lists (OpenRouter 413, Featherless up to 500).
MAX_MODELS_PER_PROFILE = 600
MAX_MODEL_ID_CHARS = 512
MAX_MODEL_LABEL_CHARS = 128
MAX_NOTE_CHARS = 512

_MODEL_ID_BLOCKLIST = re.compile(
    r"[\x00-\x1f\x7f-\x9f]"
)

_ERRORS: list[str] = []


def _error(message: str) -> None:
    _ERRORS.append(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _error(message)


def _validate_model_entry(profile_name: str, index: int, entry: object) -> None:
    if not isinstance(entry, dict):
        _error(f"profile {profile_name!r}: models[{index}] must be an object.")
        return
    model_id = entry.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        _error(f"profile {profile_name!r}: models[{index}].id must be a non-empty string.")
    else:
        model_id = model_id.strip()
        if model_id != entry["id"]:
            _error(f"profile {profile_name!r}: models[{index}].id has leading/trailing whitespace.")
        if len(model_id) > MAX_MODEL_ID_CHARS:
            _error(f"profile {profile_name!r}: models[{index}].id exceeds {MAX_MODEL_ID_CHARS} characters.")
        if _MODEL_ID_BLOCKLIST.search(model_id):
            _error(f"profile {profile_name!r}: models[{index}].id contains control characters.")

    label = entry.get("label")
    if not isinstance(label, str) or not label.strip():
        _error(f"profile {profile_name!r}: models[{index}].label must be a non-empty string.")
    elif len(label) > MAX_MODEL_LABEL_CHARS:
        _error(f"profile {profile_name!r}: models[{index}].label exceeds {MAX_MODEL_LABEL_CHARS} characters.")

    if "vision" in entry and not isinstance(entry["vision"], bool):
        _error(f"profile {profile_name!r}: models[{index}].vision must be a boolean.")

    if "notes" in entry:
        if not isinstance(entry["notes"], str):
            _error(f"profile {profile_name!r}: models[{index}].notes must be a string.")
        elif len(entry["notes"]) > MAX_NOTE_CHARS:
            _error(f"profile {profile_name!r}: models[{index}].notes exceeds {MAX_NOTE_CHARS} characters.")


def _validate_profile(profile_name: str, raw: object) -> None:
    if not isinstance(raw, dict):
        _error(f"profile {profile_name!r} must be an object.")
        return
    provider = raw.get("provider")
    _require(
        isinstance(provider, str) and provider.strip(),
        f"profile {profile_name!r}: missing non-empty 'provider'.",
    )
    models = raw.get("models")
    if not isinstance(models, list):
        _error(f"profile {profile_name!r}: 'models' must be a list.")
        return
    if len(models) > MAX_MODELS_PER_PROFILE:
        _error(f"profile {profile_name!r}: more than {MAX_MODELS_PER_PROFILE} models; is this intentional?")
    seen_ids: set[str] = set()
    for index, entry in enumerate(models):
        _validate_model_entry(profile_name, index, entry)
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            model_id = entry["id"].strip()
            if model_id in seen_ids:
                _error(f"profile {profile_name!r}: duplicate model id {model_id!r}.")
            seen_ids.add(model_id)


def main() -> int:
    try:
        with CATALOG_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {CATALOG_PATH}: {exc}", file=sys.stderr)
        return 1

    if not isinstance(payload, dict):
        print(f"ERROR: {CATALOG_PATH} must contain a JSON object.", file=sys.stderr)
        return 1

    version = payload.get("version")
    if version != 1:
        _error(f"catalog 'version' must be 1, got {version!r}.")

    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at.strip():
        _error("catalog is missing 'updated_at'.")

    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        _error("catalog is missing 'profiles' object.")
        profiles = {}

    for required_key, expected_provider in REQUIRED_PROFILE_KEYS.items():
        if required_key not in profiles:
            _error(f"missing required profile {required_key!r} for provider {expected_provider!r}.")
            continue
        raw = profiles[required_key]
        _validate_profile(required_key, raw)
        if isinstance(raw, dict):
            actual_provider = (raw.get("provider") or "").strip()
            if actual_provider != expected_provider:
                _error(
                    f"profile {required_key!r}: provider {actual_provider!r} "
                    f"does not match expected {expected_provider!r}.",
                )

    for profile_name, raw in profiles.items():
        if profile_name not in REQUIRED_PROFILE_KEYS:
            _validate_profile(profile_name, raw)

    if _ERRORS:
        print(f"ERROR: {CATALOG_PATH} failed validation:", file=sys.stderr)
        for message in _ERRORS:
            print(f"  - {message}", file=sys.stderr)
        return 1

    model_count = sum(
        len(p.get("models") or []) for p in profiles.values() if isinstance(p, dict)
    )
    print(f"OK: {CATALOG_PATH} valid ({len(profiles)} profiles, {model_count} models).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
