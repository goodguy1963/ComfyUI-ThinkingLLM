"""Persistent prompt cache and per-node workflow state shared by all backends."""

import hashlib
import json
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
PROMPT_CACHE: dict = {}
CACHE_FILE = PLUGIN_ROOT / "prompt_cache.json"
NODE_PROMPT_STATE: dict = {}
NODE_STATE_FILE = PLUGIN_ROOT / "node_prompt_state.json"


def _build_workflow_fingerprint(extra_pnginfo):
    """Return a short stable fingerprint for a workflow dict so node state
    can be scoped to a particular workflow window, avoiding cross-workflow
    leakage while remaining stable across save/reload cycles."""
    if not isinstance(extra_pnginfo, dict):
        return None
    workflow = extra_pnginfo.get("workflow")
    if not isinstance(workflow, dict):
        return None
    try:
        last_id = workflow.get("last_node_id", 0)
        node_ids = sorted(
            str(n.get("id", "")) for n in workflow.get("nodes", []) if isinstance(n, dict)
        )
        seed = f"{last_id}|{','.join(node_ids[:50])}"
        return hashlib.md5(seed.encode()).hexdigest()[:12]
    except Exception:
        return None

def _make_node_state_key(node_class, unique_id, extra_pnginfo):
    """Produce a stable per-node key from the node class name, its numeric
    identity within the workflow, and the workflow fingerprint when available."""
    wf_fp = _build_workflow_fingerprint(extra_pnginfo)
    if wf_fp:
        return f"{node_class}|{wf_fp}|{unique_id}"
    return f"{node_class}|{unique_id}"


def _normalize_state_signature_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_state_signature_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_state_signature_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return str(value)


def build_node_input_signature(**kwargs):
    """Return a stable hash for the node inputs that should invalidate saved prompt reuse."""
    normalized = {
        str(key): _normalize_state_signature_value(value)
        for key, value in sorted(kwargs.items(), key=lambda item: str(item[0]))
    }
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def load_node_prompt_state():
    """Load per-node prompt state from the sidecar JSON file."""
    try:
        if NODE_STATE_FILE.exists():
            with open(NODE_STATE_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError("node prompt state must be a JSON object")
                NODE_PROMPT_STATE.clear()
                NODE_PROMPT_STATE.update(loaded)
                print(f"[QwenVL] Loaded {len(NODE_PROMPT_STATE)} node prompt states")
    except Exception as e:
        print(f"[QwenVL] Failed to load node prompt state: {e}")
        NODE_PROMPT_STATE.clear()

def save_node_prompt_state():
    """Persist per-node prompt state to the sidecar JSON file."""
    try:
        with open(NODE_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(NODE_PROMPT_STATE, f, indent=2)
    except Exception as e:
        print(f"[QwenVL] Failed to save node prompt state: {e}")

def get_node_saved_prompt(node_class, unique_id, extra_pnginfo):
    """Return the saved prompt text for a specific node, or None."""
    key = _make_node_state_key(node_class, unique_id, extra_pnginfo)
    entry = NODE_PROMPT_STATE.get(key)
    if isinstance(entry, dict) and entry.get("text"):
        return entry["text"]
    return None


def get_node_saved_prompt_with_seed(node_class, unique_id, extra_pnginfo, seed=None, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None, input_signature=None):
    """Return saved prompt text only if the generation parameters haven't changed."""
    key = _make_node_state_key(node_class, unique_id, extra_pnginfo)
    entry = NODE_PROMPT_STATE.get(key)
    if not isinstance(entry, dict) or not entry.get("text"):
        return None
    # If seed or params changed, invalidate and return None
    if seed is not None and entry.get("seed") != seed:
        return None
    if max_tokens is not None and entry.get("max_tokens") != max_tokens:
        return None
    if temperature is not None and entry.get("temperature") != temperature:
        return None
    if top_p is not None and entry.get("top_p") != top_p:
        return None
    if repetition_penalty is not None and entry.get("repetition_penalty") != repetition_penalty:
        return None
    if input_signature is not None and entry.get("input_signature") != input_signature:
        return None
    return entry["text"]

def set_node_saved_prompt(node_class, unique_id, extra_pnginfo, text, raw_trace=None, seed=None, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None, input_signature=None):
    """Persist the generated prompt (and optional raw trace) for a specific node."""
    key = _make_node_state_key(node_class, unique_id, extra_pnginfo)
    NODE_PROMPT_STATE[key] = {
        "text": text,
        "raw_trace": raw_trace or "",
        "timestamp": None,
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "input_signature": input_signature,
    }
    save_node_prompt_state()

def load_prompt_cache():
    """Load prompt cache from file"""
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError("prompt cache must be a JSON object")
                PROMPT_CACHE.clear()
                PROMPT_CACHE.update(loaded)
                print(f"[QwenVL] Loaded {len(PROMPT_CACHE)} cached prompts")
    except Exception as e:
        print(f"[QwenVL] Failed to load prompt cache: {e}")
        PROMPT_CACHE.clear()

def save_prompt_cache():
    """Save prompt cache to file"""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(PROMPT_CACHE, f, indent=2)
    except Exception as e:
        print(f"[QwenVL] Failed to save prompt cache: {e}")


def log_llm_input(
    backend: str,
    stage: str,
    preset_name: str,
    user_text: str,
    *,
    system_text: str = "",
    formatted_text: str | None = None,
    media: dict[str, int] | None = None,
) -> None:
    """Print the complete text input sent to an LLM without dumping media payloads."""
    lines = [
        f"\n[{backend}] LLM INPUT",
        f"Stage: {stage}",
        f"Preset: {preset_name or '(none)'}",
        "Media: " + (", ".join(f"{kind}={count}" for kind, count in (media or {}).items()) or "none"),
        "--- SYSTEM ---",
        system_text or "(none)",
        "--- USER ---",
        user_text or "(empty)",
    ]
    if formatted_text is not None:
        lines.extend(("--- FORMATTED CHAT TEMPLATE ---", formatted_text))
    lines.append("--- END LLM INPUT ---")
    message = "\n".join(lines)
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(message.encode(encoding, errors="replace").decode(encoding))

def get_cache_key(model_name, preset_prompt, custom_prompt, image_hash=None, video_hash=None, seed=None, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None, enable_thinking=None, effective_duration_seconds=None, video_prompt_contract_version=None):
    """Generate cache key from inputs including all generation parameters.

    All generation-parameter fields are optional so existing callers that omit them
    continue to work; when provided, changing any parameter produces a different key.
    """
    key_data = {
        "model": model_name,
        "preset": preset_prompt,
        "custom": custom_prompt.strip() if custom_prompt else "",
        "image": image_hash,
        "video": video_hash,
        "seed": seed,  # Always include seed to ensure proper caching behavior
        "max_tokens": max_tokens,
        "temperature": temperature if temperature is not None else None,
        "top_p": top_p if top_p is not None else None,
        "repetition_penalty": repetition_penalty if repetition_penalty is not None else None,
        "enable_thinking": bool(enable_thinking) if enable_thinking is not None else None,
    }
    if effective_duration_seconds is not None:
        key_data["effective_duration_seconds"] = effective_duration_seconds
    if video_prompt_contract_version is not None:
        key_data["video_prompt_contract_version"] = video_prompt_contract_version
    # Create deterministic hash
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def get_alternative_cache_key(model_name, preset_prompt, custom_prompt, image_hash=None, video_hash=None, seed=None, module_name="QwenVL"):
    """Generate alternative cache key for fixed seed mode to find random prompts"""
    # Only for fixed seed mode (when user wants consistent prompts)
    # We consider any seed that the user keeps fixed as "fixed seed mode"

    print(f"[{module_name} DEBUG] Searching through cache for model={model_name}, preset={preset_prompt}")

    # Try to find any cached prompt with same model/preset/custom/image but different seed
    for cached_key, cached_data in PROMPT_CACHE.items():
        cached_model = cached_data.get("model")
        cached_preset = cached_data.get("preset")
        cached_seed = cached_data.get("seed")

        print(f"[{module_name} DEBUG] Checking entry: model={cached_model}, preset={cached_preset}, seed={cached_seed}")

        if (cached_model == model_name and
            cached_preset == preset_prompt and
            cached_seed != seed):  # Different seed

            # Generate the cache key that would have been created for this cached data
            # to check if image/video hashes match
            cached_image_hash = cached_data.get("image_hash")
            cached_video_hash = cached_data.get("video_hash")

            print(f"[{module_name} DEBUG] Found potential match with hashes: image={cached_image_hash}, video={cached_video_hash}")

            # If the cached data doesn't have hash info, try to match by other criteria
            if cached_image_hash is None and cached_video_hash is None:
                # Fallback: if both current and cached have no image/video, consider it a match
                if image_hash is None and video_hash is None:
                    print(f"[{module_name} DEBUG] Match found (no images/videos)!")
                    return cached_key
            else:
                # Match if hashes are the same (including None)
                if cached_image_hash == image_hash and cached_video_hash == video_hash:
                    print(f"[{module_name} DEBUG] Match found (hashes match)!")
                    return cached_key
    print(f"[{module_name} DEBUG] No alternative cache found")
    return None
