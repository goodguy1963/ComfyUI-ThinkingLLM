"""Fetch live provider model lists and regenerate web/api_model_catalogs.json.

For providers that expose a public (key-less) OpenAI-compatible /models
endpoint we fetch the full list and refresh the catalog automatically. For
providers that require an API key (OpenAI, QwenCloud, Groq, Together AI,
Fireworks) the curated entries in web/api_model_catalogs.json are kept as-is.

Usage:
    python tools/update_api_model_catalogs.py            # fetch public providers
    python tools/update_api_model_catalogs.py --offline  # keep current data
    python tools/update_api_model_catalogs.py --dry-run  # print what would change

After running, run the validator:
    python tools/check_api_model_catalogs.py

Run from the repository root. This is intended to run in CI at release time
so the shipped lists stay current.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "web" / "api_model_catalogs.json"

# Providers with a public, key-less /models endpoint.
PUBLIC_PROVIDERS: dict[str, str] = {
    "OpenRouter": "https://openrouter.ai/api/v1/models",
    "OrcaRouter": "https://api.orcarouter.ai/v1/models",
    "DeepInfra": "https://api.deepinfra.com/v1/openai/models",
    "Featherless": "https://api.featherless.ai/v1/models",
}

# Providers that require a key and are therefore hand-curated.
CURATED_PROVIDERS = ["OpenAI", "QwenCloud (Singapore)", "Together AI", "Fireworks AI", "Groq"]

# Maximum number of entries kept per provider (to keep the node dropdown usable).
MAX_MODELS_PER_PROVIDER = 500


def _fetch_json(url: str, timeout: int = 30) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _extract_models_openrouter(payload: dict) -> list[str]:
    items = payload.get("data") or []
    # Sort by context_length descending so the most capable models come first.
    items.sort(key=lambda m: m.get("context_length") or 0, reverse=True)
    return [m["id"] for m in items if isinstance(m.get("id"), str)]


def _extract_models_openai_style(payload: dict | list) -> list[str]:
    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("data") or payload.get("models") or []
    ids = [m.get("id") for m in items if isinstance(m, dict)]
    return [i for i in ids if isinstance(i, str)]


def _filter_featherless(ids: list[str]) -> list[str]:
    """Featherless exposes thousands of raw HF names; keep the usable chat/vision ones."""
    keywords = ("instruct", "chat", "vision", "vl", "turbo", "-it", "1.5", "3.5")
    blocked = ("gensyn-swarm", "-gensyn-", "swarm-")
    out = [
        i for i in ids
        if any(k.lower() in i.lower() for k in keywords)
        and not any(b.lower() in i.lower() for b in blocked)
    ]
    # De-dup case-insensitively, keep order.
    seen: set[str] = set()
    result: list[str] = []
    for model_id in out:
        key = model_id.lower()
        if key not in seen:
            seen.add(key)
            result.append(model_id)
    return result


def _sort_models_for_dropdown(ids: list[str], *, by_context: bool = False,
                              context_map: dict[str, int] | None = None) -> list[str]:
    """Sort for the dropdown: real models first, special router models last.

    When by_context is set (OpenRouter), order real models by context length
    descending so the most capable models appear first.
    """
    def sort_key(model_id: str):
        lowered = model_id.lower()
        is_special = lowered.startswith("orcarouter/") or lowered in {"openrouter/auto", "openrouter/auto-beta"}
        ctx = -(context_map.get(model_id, 0) or 0) if by_context else 0
        return (1 if is_special else 0, ctx, lowered)
    return sorted(ids, key=sort_key)


def _fetch_public_catalog() -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for provider, url in PUBLIC_PROVIDERS.items():
        try:
            payload = _fetch_json(url)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[update] WARNING: could not fetch {provider}: {exc}", file=sys.stderr)
            continue

        if provider == "OpenRouter":
            ids = _extract_models_openrouter(payload)
            context_map = {
                m["id"]: (m.get("context_length") or 0)
                for m in (payload.get("data") or []) if isinstance(m, dict) and isinstance(m.get("id"), str)
            }
        elif provider == "Featherless":
            ids = _filter_featherless(_extract_models_openai_style(payload))
            context_map = {}
        else:
            ids = _extract_models_openai_style(payload)
            context_map = {}

        if not ids:
            print(f"[update] WARNING: {provider} returned no models", file=sys.stderr)
            continue

        ids = _sort_models_for_dropdown(ids, by_context=(provider == "OpenRouter"), context_map=context_map)
        models = [
            {"id": model_id, "label": model_id, "vision": False}
            for model_id in ids[:MAX_MODELS_PER_PROVIDER]
        ]
        catalog[provider] = {"provider": provider, "models": models}
        print(f"[update] {provider}: {len(ids)} models (kept {len(models)}).")
    return catalog


def _merge_catalogs(previous: dict, fresh: dict) -> dict:
    """Replace public providers with fresh data, keep curated providers as-is."""
    profiles = dict(previous.get("profiles", {}))
    for provider, entry in fresh.items():
        profiles[provider] = entry
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser(description="Update the API model catalogs.")
    parser.add_argument("--offline", action="store_true", help="Do not fetch; just rewrite metadata.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing.")
    args = parser.parse_args()

    if not CATALOG_PATH.exists():
        print(f"ERROR: {CATALOG_PATH} not found.", file=sys.stderr)
        return 1
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        previous = json.load(handle)

    fresh = {} if args.offline else _fetch_public_catalog()
    profiles = _merge_catalogs(previous, fresh)
    now = date.today().isoformat()

    result = {
        "version": 1,
        "description": previous.get(
            "description",
            "Curated convenience model lists for the ThinkingLLM API node. "
            "These are suggestions only and never override a server-side profile allowed_models allowlist.",
        ),
        "updated_at": now,
        "profiles": profiles,
    }

    if args.dry_run:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    with CATALOG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[update] Wrote {CATALOG_PATH} (updated_at={now}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
