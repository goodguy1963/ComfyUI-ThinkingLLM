# ComfyUI-QwenVL
# This custom node integrates the Qwen-VL series, including the latest Qwen3-VL models,
# including Qwen2.5-VL and the latest Qwen3-VL, to enable advanced multimodal AI for text generation,
# image understanding, and video analysis.
#
# Models License Notice:
# - Qwen3-VL: Apache-2.0 License (https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
# - Qwen2.5-VL: Apache-2.0 License (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL

import gc
import inspect
import json
import os
import platform
import shutil
import time
from enum import Enum
from pathlib import Path
import hashlib

import numpy as np
import psutil
import torch
from PIL import Image
try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    from huggingface_hub import snapshot_download
    hf_hub_download = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
try:
    from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    AutoProcessor = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
from AILab_StreamDisplay import TerminalStreamDisplay
from comfy.model_management import throw_exception_if_processing_interrupted

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None

try:
    from server import PromptServer
except Exception:
    PromptServer = None

# SageAttention support
try:
    from sageattention.core import (
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
    )
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False

# Global cache for generated prompts
PROMPT_CACHE = {}
CACHE_FILE = Path(__file__).parent / "prompt_cache.json"

# DEPRECATED: per-node state via set_node_saved_prompt / get_node_saved_prompt is used instead.
LAST_SAVED_PROMPT = None  # kept only to avoid ImportError in legacy importers

# Per-node prompt state: survives idle, module reload, ComfyUI restart.
# Keyed by node identity (workflow fingerprint + unique_id + node class).
NODE_PROMPT_STATE = {}
NODE_STATE_FILE = Path(__file__).parent / "node_prompt_state.json"
QWEN_MIN_REASONING_BUDGET_TOKENS = 1024
QWEN_FINAL_ANSWER_RESERVE_TOKENS = 256
QWEN_CONTEXT_OVERHEAD_TOKENS = 128
_QWEN_SOFT_SWITCHES = {"/think", "/no_think", "/nothink"}
_STREAM_HEARTBEAT_INTERVAL_SECONDS = 3.0
_DOWNLOAD_STATUS_INTERVAL_SECONDS = 0.25
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _format_download_size(num_bytes) -> str:
    if num_bytes is None:
        return "? B"
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


class _DownloadProgressReporter:
    def __init__(self, node_id=None, label: str = "QwenVL Download", repo_id: str | None = None):
        self.node_id = node_id
        self.label = label
        self.repo_id = repo_id
        self.progress_bar = ProgressBar(1, node_id=node_id) if ProgressBar is not None else None
        self._last_text_at = 0.0
        self._last_terminal_at = 0.0
        self._last_percent = None

    def _emit_ui_text(self, text: str, *, force: bool = False) -> None:
        if PromptServer is None or self.node_id is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_text_at) < _DOWNLOAD_STATUS_INTERVAL_SECONDS:
            return
        PromptServer.instance.send_progress_text(text, self.node_id)
        self._last_text_at = now

    def _emit_terminal_text(self, text: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_terminal_at) < _DOWNLOAD_STATUS_INTERVAL_SECONDS:
            return
        print(text)
        self._last_terminal_at = now

    def _build_message(self, stage: str, detail: str | None = None, current: int | float | None = None, total: int | float | None = None) -> str:
        lines = [self.label, f"Stage: {stage}"]
        if self.repo_id:
            lines.append(f"Source: {self.repo_id}")
        if detail:
            lines.append(f"File: {detail}")
        if total and total > 0:
            percent = int(min(100, max(0, round((float(current or 0) / float(total)) * 100))))
            lines.append(f"Progress: {percent}% ({_format_download_size(current)} / {_format_download_size(total)})")
        elif current is not None:
            lines.append(f"Downloaded: {_format_download_size(current)}")
        return "\n".join(lines)

    def stage(self, stage: str, *, detail: str | None = None, force: bool = True) -> None:
        message = self._build_message(stage, detail=detail)
        self._emit_ui_text(message, force=force)
        self._emit_terminal_text(f"[{self.label}] {stage}{f' - {detail}' if detail else ''}", force=force)

    def progress(self, stage: str, *, detail: str | None = None, current: int | float | None = None, total: int | float | None = None, force: bool = False) -> None:
        if self.progress_bar is not None and total is not None and total > 0:
            self.progress_bar.update_absolute(min(float(current or 0), float(total)), total=float(total))
        percent = None
        if total and total > 0:
            percent = int(min(100, max(0, round((float(current or 0) / float(total)) * 100))))
        message = self._build_message(stage, detail=detail, current=current, total=total)
        self._emit_ui_text(message, force=force or percent != self._last_percent)
        suffix = ""
        if percent is not None:
            suffix = f" {percent}% ({_format_download_size(current)} / {_format_download_size(total)})"
        elif current is not None:
            suffix = f" {_format_download_size(current)}"
        self._emit_terminal_text(f"[{self.label}] {stage}{f' - {detail}' if detail else ''}{suffix}", force=force or percent != self._last_percent)
        self._last_percent = percent

    def finish(self, *, detail: str | None = None, force: bool = True) -> None:
        if self.progress_bar is not None:
            self.progress_bar.update_absolute(1, total=1)
        message = self._build_message("Completed", detail=detail, current=1, total=1)
        self._emit_ui_text(message, force=force)
        self._emit_terminal_text(f"[{self.label}] Completed{f' - {detail}' if detail else ''}", force=force)

    def fail(self, detail: str) -> None:
        message = self._build_message("Failed", detail=detail)
        self._emit_ui_text(message, force=True)
        self._emit_terminal_text(f"[{self.label}] Failed - {detail}", force=True)

    def make_tqdm_class(self):
        if tqdm is None:
            return None
        reporter = self

        class _NodeDownloadTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                reporter.progress(
                    "Downloading",
                    detail=getattr(self, "desc", None),
                    current=getattr(self, "n", 0),
                    total=getattr(self, "total", None),
                    force=True,
                )

            def update(self, n=1):
                result = super().update(n)
                reporter.progress(
                    "Downloading",
                    detail=getattr(self, "desc", None),
                    current=getattr(self, "n", 0),
                    total=getattr(self, "total", None),
                )
                return result

            def refresh(self, *args, **kwargs):
                result = super().refresh(*args, **kwargs)
                reporter.progress(
                    "Downloading",
                    detail=getattr(self, "desc", None),
                    current=getattr(self, "n", 0),
                    total=getattr(self, "total", None),
                )
                return result

            def close(self):
                reporter.progress(
                    "Downloading",
                    detail=getattr(self, "desc", None),
                    current=(getattr(self, "total", None) or getattr(self, "n", 0)),
                    total=(getattr(self, "total", None) or getattr(self, "n", 0) or None),
                    force=True,
                )
                return super().close()

        return _NodeDownloadTqdm


def _maybe_emit_hf_stream_heartbeat(stage_label: str, started_at: float, last_status_at: float, full_text: str) -> float:
    now = time.monotonic()
    if (now - last_status_at) < _STREAM_HEARTBEAT_INTERVAL_SECONDS:
        return last_status_at
    elapsed = max(0.0, now - started_at)
    print(f"[QwenVL HF] {stage_label}: still generating... ({len(full_text)} chars, {elapsed:.1f}s)")
    return now

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
    global NODE_PROMPT_STATE
    try:
        if NODE_STATE_FILE.exists():
            with open(NODE_STATE_FILE, "r", encoding="utf-8") as f:
                NODE_PROMPT_STATE = json.load(f)
                print(f"[QwenVL] Loaded {len(NODE_PROMPT_STATE)} node prompt states")
    except Exception as e:
        print(f"[QwenVL] Failed to load node prompt state: {e}")
        NODE_PROMPT_STATE = {}

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
    global PROMPT_CACHE
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                PROMPT_CACHE = json.load(f)
                print(f"[QwenVL] Loaded {len(PROMPT_CACHE)} cached prompts")
    except Exception as e:
        print(f"[QwenVL] Failed to load prompt cache: {e}")
        PROMPT_CACHE = {}

def save_prompt_cache():
    """Save prompt cache to file"""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(PROMPT_CACHE, f, indent=2)
    except Exception as e:
        print(f"[QwenVL] Failed to save prompt cache: {e}")


def estimate_qwen_text_tokens(*parts):
    """Conservatively estimate prompt tokens when an exact tokenizer is unavailable."""
    total_chars = 0
    non_empty_parts = 0
    for part in parts:
        if not part:
            continue
        normalized = " ".join(str(part).split())
        if not normalized:
            continue
        total_chars += len(normalized)
        non_empty_parts += 1
    if total_chars <= 0:
        return 0
    return max(1, (total_chars + 3) // 4 + max(0, non_empty_parts - 1))


def resolve_qwen_context_window(config, fallback=32768):
    """Return the best available HF context window from the model config."""
    candidates = []
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is None:
            continue
        for attr in ("max_position_embeddings", "n_positions", "seq_length"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and 0 < value <= 2_000_000:
                candidates.append(value)
    if candidates:
        return max(candidates)
    return int(fallback)


def resolve_qwen_thinking_mode(
    requested_enable,
    max_tokens,
    label="QwenVL",
    *,
    prompt_tokens=None,
    context_window=None,
    min_reasoning_tokens=QWEN_MIN_REASONING_BUDGET_TOKENS,
    answer_reserve_tokens=QWEN_FINAL_ANSWER_RESERVE_TOKENS,
    context_overhead_tokens=QWEN_CONTEXT_OVERHEAD_TOKENS,
):
    """Enable thinking only when requested and enough generation budget remains."""
    if not bool(requested_enable):
        return False
    requested_output = max(0, int(max_tokens or 0))
    available_output = requested_output
    prompt_token_count = max(0, int(prompt_tokens or 0))
    context_limit = int(context_window) if context_window is not None else None

    if context_limit is not None:
        remaining_context = max(0, context_limit - prompt_token_count - int(context_overhead_tokens))
        available_output = min(available_output, remaining_context) if available_output else remaining_context

    minimum_required = int(min_reasoning_tokens) + int(answer_reserve_tokens)
    if available_output <= 0 or available_output < minimum_required:
        if context_limit is not None:
            budget_note = (
                f"max_tokens={requested_output}, prompt_tokens={prompt_token_count}, "
                f"context_window={context_limit}, available_output={available_output}"
            )
        else:
            budget_note = f"max_tokens={requested_output}, available_output={available_output}"
        print(
            f"[{label}] Thinking requested but {budget_note} is below the required {minimum_required} tokens "
            f"({int(min_reasoning_tokens)} reasoning + {int(answer_reserve_tokens)} answer reserve); "
            "forcing non-thinking mode to preserve answer space."
        )
        return False
    return True


def apply_qwen_soft_thinking_directive(prompt_text, enable_thinking, supports_soft_switch=False):
    """Append the latest Qwen3 soft switch when the backend supports it."""
    text = prompt_text or ""
    if not supports_soft_switch:
        return text
    lines = [line for line in text.splitlines() if line.strip().lower() not in _QWEN_SOFT_SWITCHES]
    lines.append("/think" if enable_thinking else "/no_think")
    return "\n".join(lines).strip()

def get_cache_key(model_name, preset_prompt, custom_prompt, image_hash=None, video_hash=None, seed=None, max_tokens=None, temperature=None, top_p=None, repetition_penalty=None, enable_thinking=None):
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

def ensure_cuda_vram_headroom(module_name="QwenVL", min_free_gb=1.0, min_free_ratio=0.08):
    if not torch.cuda.is_available():
        return True
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        free_before, total = torch.cuda.mem_get_info()
    except Exception:
        gc.collect()
        torch.cuda.empty_cache()
        return True

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    reclaimable = max(reserved - allocated, 0)
    threshold = max(int(min_free_gb * 1024**3), int(total * min_free_ratio))

    if free_before >= threshold and reclaimable < 512 * 1024**2:
        return True

    print(
        f"[{module_name}] VRAM headroom low before run: "
        f"free={free_before / 1024**3:.2f}GB, "
        f"reserved={reserved / 1024**3:.2f}GB, "
        f"allocated={allocated / 1024**3:.2f}GB. Cleaning CUDA cache..."
    )
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    try:
        free_after, _ = torch.cuda.mem_get_info()
        print(f"[{module_name}] VRAM after cleanup: free={free_after / 1024**3:.2f}GB")
        return free_after >= threshold
    except Exception:
        return True

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image with memory optimization"""
    if tensor is None:
        return None
    try:
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        # More aggressive memory management
        if tensor.is_floating_point():
            # Scale to 0-255 range more efficiently
            tensor = tensor.clamp(0, 1)
        
        # Reduce sample size for memory efficiency
        array = tensor.cpu().numpy()
        if tensor.numel() > 0:
            # Take smaller sample for hash to save memory
            sample_size = min(50, tensor.numel() // 4)  # Reduced from 100
            sample_pixels = array.flatten()[:sample_size].tolist() if array.size > 0 else []
        else:
            sample_pixels = []
        
        content = f"{tensor.shape}_{tensor.dtype}_{sample_pixels[:10]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except:
        return None

def get_image_hash(image):
    """Generate hash for image tensor"""
    if image is None:
        return None
    try:
        # Use image tensor properties for hash
        shape = str(image.shape)
        dtype = str(image.dtype)
        # Sample a few pixels for content hash (avoid full tensor for performance)
        if len(image.shape) >= 3:
            sample_pixels = image.flatten()[:100].tolist() if image.numel() > 0 else []
        else:
            sample_pixels = image.flatten().tolist() if image.numel() > 0 else []
        
        content = f"{shape}_{dtype}_{sample_pixels[:10]}"  # Limit sample size
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except:
        return None

def get_video_hash(video):
    """Generate hash for video tensor (same as image)"""
    return get_image_hash(video)

# Load cache and per-node state on module import
load_prompt_cache()
load_node_prompt_state()
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

# Export memory functions for external use
__all__ = [
    'PROMPT_CACHE', 'NODE_PROMPT_STATE',
    'get_cache_key', 'get_alternative_cache_key',
    'save_prompt_cache',
    'get_image_hash', 'get_video_hash',
    'check_pytorch_memory', 'set_pytorch_memory_fraction',
    'get_device_info', 'tensor_to_pil', 'enforce_memory',
    'quantization_config', 'ensure_model', 'resolve_attention_mode',
    'flash_attn_available', 'normalize_device_choice',
    'load_model_configs',
    'HF_VL_MODELS', 'HF_TEXT_MODELS', 'HF_ALL_MODELS',
    'SYSTEM_PROMPTS', 'PRESET_PROMPTS', 'TOOLTIPS',
    'Quantization', 'ATTENTION_MODES',
    'NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS',
    'get_node_saved_prompt_with_seed', 'get_node_saved_prompt', 'set_node_saved_prompt',
    'load_node_prompt_state', 'save_node_prompt_state',
    '_make_node_state_key',
]

import folder_paths

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "hf_models.json"
SYSTEM_PROMPTS_PATH = NODE_DIR / "AILab_System_Prompts.json"
HF_VL_MODELS: dict[str, dict] = {}
HF_TEXT_MODELS: dict[str, dict] = {}
HF_ALL_MODELS: dict[str, dict] = {}
SYSTEM_PROMPTS = {}
NO_PRESET_PROMPT = "🚫 No preset (image-only)"
PRESET_PROMPTS: list[str] = [NO_PRESET_PROMPT, "Describe this image in detail."]

TOOLTIPS = {
    "model_name": "Pick the Qwen-VL checkpoint. First run downloads weights into models/LLM/Qwen-VL, so leave disk space.",
    "quantization": "Precision vs VRAM. FP16 gives the best quality if memory allows; 8-bit suits 8–16 GB GPUs; 4-bit fits 6 GB or lower but is slower.",
    "attention_mode": "auto tries SageAttention → FlashAttention 2 → SDPA in order. SDPA is stable and recommended. Only override when debugging attention backends.",
    "preset_prompt": "Built-in instruction describing how Qwen-VL should analyze the media input.",
    "custom_prompt": "Additional user input that gets combined with the preset template. Leave empty to use only the template.",
    "max_tokens": "Maximum number of new tokens to decode. Larger values yield longer answers but consume more time and memory.",
    "keep_model_loaded": "Keeps the model resident in VRAM/RAM after the run so the next prompt skips loading.",
    "seed": "Seed controlling sampling and frame picking; reuse it to reproduce results.",
    "use_torch_compile": "Enable torch.compile('reduce-overhead') on supported CUDA/Torch 2.1+ builds for extra throughput after the first compile.",
    "device": "Choose where to run the model: auto, cpu, mps, or cuda:x for multi-GPU systems.",
    "temperature": "Sampling randomness when num_beams == 1. 0.2–0.4 is focused, 0.7+ is creative.",
    "top_p": "Nucleus sampling cutoff when num_beams == 1. Lower values keep only top tokens; 0.9–0.95 allows more variety.",
    "num_beams": "Beam-search width. Values >1 disable temperature/top_p and trade speed for more stable answers.",
    "repetition_penalty": "Values >1 (e.g., 1.1–1.3) penalize repeated phrases; 1.0 leaves logits untouched.",
    "frame_count": "Number of frames extracted from video inputs before prompting Qwen-VL. More frames provide context but cost time.",
    "hf_token": "Optional Hugging Face access token for private or gated model downloads. It is passed only to the download call, never logged or cached, and the in-memory copy is dropped after the download attempt. Clear this field before saving or sharing workflows.",
}


def _iter_hf_token_candidates(hf_token: str | None):
    token = str(hf_token or "").strip()
    if token:
        yield token
    for env_name in _HF_TOKEN_ENV_VARS:
        token = str(os.environ.get(env_name) or "").strip()
        if token:
            yield token


def _clean_hf_token(hf_token: str | None) -> str | None:
    return next(_iter_hf_token_candidates(hf_token), None)


def _redact_hf_token(text: str, hf_token: str | None) -> str:
    redacted = str(text)
    for token in _iter_hf_token_candidates(hf_token):
        redacted = redacted.replace(token, "<redacted HF token>")
    return redacted


def _hf_download_error_message(exc: Exception, *, repo_id: str, hf_token: str | None) -> str:
    message = _redact_hf_token(str(exc), hf_token)
    lower = message.lower()
    access_markers = (
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "gated",
        "private",
        "repository not found",
        "requires authentication",
    )
    if any(marker in lower for marker in access_markers):
        if _clean_hf_token(hf_token):
            message += (
                f"\n[QwenVL] Hugging Face access hint for {repo_id}: a token was supplied but access was rejected. "
                "Confirm the token has read permission and that you accepted the model license/gate on Hugging Face."
            )
        else:
            message += (
                f"\n[QwenVL] Hugging Face access hint for {repo_id}: this model may be private or gated. "
                "Paste a read token into the hf_token field for this run, or set up access on Hugging Face first."
            )
    return message


def _call_hf_hub_download(*, repo_id: str, filename: str, local_dir: Path, token: str | None, reporter: _DownloadProgressReporter) -> Path:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub.hf_hub_download is unavailable in this environment")
    download_kwargs = {
        "repo_id": repo_id,
        "repo_type": "model",
        "filename": filename,
        "local_dir": str(local_dir),
    }
    if token:
        download_kwargs["token"] = token
    tqdm_cls = reporter.make_tqdm_class()
    if tqdm_cls is not None:
        try:
            if "tqdm_class" in inspect.signature(hf_hub_download).parameters:
                download_kwargs["tqdm_class"] = tqdm_cls
        except (TypeError, ValueError):
            pass
    try:
        return Path(hf_hub_download(**download_kwargs))
    finally:
        download_kwargs.pop("token", None)

class Quantization(str, Enum):
    Q4 = "4-bit (VRAM-friendly)"
    Q8 = "8-bit (Balanced)"
    FP16 = "None (FP16)"

    @classmethod
    def get_values(cls):
        return [item.value for item in cls]

    @classmethod
    def from_value(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"Unsupported quantization: {value}")

ATTENTION_MODES = ["auto", "sage", "flash_attention_2", "sdpa"]

# Debug: Check SageAttention availability
print(f"[QwenVL Debug] SAGE_ATTENTION_AVAILABLE: {SAGE_ATTENTION_AVAILABLE}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"[QwenVL Debug] CUDA capability: {major}.{minor}")
else:
    print("[QwenVL Debug] CUDA not available")
print(f"[QwenVL Debug] Final ATTENTION_MODES: {ATTENTION_MODES}")

# Temporarily show sage option even if not available for testing
print("[QwenVL] NOTE: SageAttention option shown for testing. Install with: pip install sageattention")

def load_model_configs():
    global HF_VL_MODELS, HF_TEXT_MODELS, HF_ALL_MODELS, SYSTEM_PROMPTS, PRESET_PROMPTS
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        if "hf_vl_models" in data or "hf_text_models" in data:
            HF_VL_MODELS = data.get("hf_vl_models") or {}
            HF_TEXT_MODELS = data.get("hf_text_models") or {}
        else:
            HF_VL_MODELS = {k: v for k, v in data.items() if not k.startswith("_")}
            HF_TEXT_MODELS = {}
        SYSTEM_PROMPTS = data.get("_system_prompts", {})
        PRESET_PROMPTS = data.get("_preset_prompts", PRESET_PROMPTS)
    except Exception as exc:
        print(f"[QwenVL] Config load failed: {exc}")
        HF_VL_MODELS = {}
        HF_TEXT_MODELS = {}
        HF_ALL_MODELS = {}
        SYSTEM_PROMPTS = {}
    try:
        with open(SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwenvl_prompts = data.get("qwenvl") or {}
        preset_override = data.get("_preset_prompts") or []
        if isinstance(qwenvl_prompts, dict) and qwenvl_prompts:
            SYSTEM_PROMPTS = qwenvl_prompts
        if isinstance(preset_override, list) and preset_override:
            PRESET_PROMPTS = preset_override
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[QwenVL] System prompts load failed: {exc}")

    if NO_PRESET_PROMPT not in PRESET_PROMPTS:
        PRESET_PROMPTS = [NO_PRESET_PROMPT, *PRESET_PROMPTS]
    if isinstance(SYSTEM_PROMPTS, dict):
        SYSTEM_PROMPTS.setdefault(NO_PRESET_PROMPT, "")
    custom = NODE_DIR / "custom_models.json"
    if custom.exists():
        try:
            with open(custom, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            custom_vl = data.get("hf_vl_models") or {}
            custom_text = data.get("hf_text_models") or {}
            legacy = data.get("hf_models", {}) or data.get("models", {})
            if isinstance(custom_vl, dict) and custom_vl:
                HF_VL_MODELS.update(custom_vl)
                print(f"[QwenVL] Loaded {len(custom_vl)} custom VL models")
            if isinstance(custom_text, dict) and custom_text:
                HF_TEXT_MODELS.update(custom_text)
                print(f"[QwenVL] Loaded {len(custom_text)} custom text models")
            if isinstance(legacy, dict) and legacy:
                HF_VL_MODELS.update(legacy)
                print(f"[QwenVL] Loaded {len(legacy)} custom legacy models")
        except Exception as exc:
            print(f"[QwenVL] custom_models.json skipped: {exc}")
    HF_ALL_MODELS = dict(HF_VL_MODELS)
    HF_ALL_MODELS.update(HF_TEXT_MODELS)

    # Scan local models directory for HF models not in JSON config
    _scan_local_hf_models()


def _scan_local_hf_models():
    """Scan all LLM/Qwen-VL directories for locally available HF models not already in the config."""
    global HF_VL_MODELS, HF_ALL_MODELS

    # Collect all Qwen-VL directories to scan from ComfyUI's multi-path system
    scan_dirs: list[Path] = []
    llm_paths = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
    for llm_path in llm_paths:
        qwen_dir = Path(llm_path) / "Qwen-VL"
        if qwen_dir not in scan_dirs:
            scan_dirs.append(qwen_dir)
    # Always include the default location as fallback
    default_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"
    if default_dir not in scan_dirs:
        scan_dirs.append(default_dir)

    # Collect known local directory names from JSON config
    known_dirs = set()
    for info in HF_ALL_MODELS.values():
        repo_id = info.get("repo_id", "")
        if isinstance(repo_id, str) and "/" in repo_id:
            known_dirs.add(repo_id.split("/")[-1])
        local_path = info.get("local_path")
        if local_path:
            known_dirs.add(Path(local_path).name)

    count = 0
    for models_dir in scan_dirs:
        if not models_dir.exists() or not models_dir.is_dir():
            continue
        try:
            for entry in models_dir.iterdir():
                if not entry.is_dir():
                    continue
                if entry.name in known_dirs:
                    continue
                # Check if the directory contains model weights
                has_weights = any(entry.glob("*.safetensors")) or any(entry.glob("*.bin"))
                if not has_weights:
                    continue
                display = f"[local] {entry.name}"
                model_info = {
                    "local_path": str(entry),
                    "repo_id": None,
                    "is_local": True,
                    "quantized": False,
                }
                HF_VL_MODELS[display] = model_info
                HF_ALL_MODELS[display] = model_info
                known_dirs.add(entry.name)
                count += 1
        except PermissionError:
            pass

    if count:
        print(f"[QwenVL] Discovered {count} local HF model(s) on disk")


if not HF_ALL_MODELS:
    load_model_configs()


def read_hf_model_type(model_dir: str) -> str | None:
    """Read model_type from a HuggingFace model's config.json."""
    try:
        config_path = Path(model_dir) / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config.get("model_type")
    except Exception:
        pass
    return None

def check_pytorch_memory():
    """Check current PyTorch memory settings and allow user to set fraction"""
    try:
        import torch
        print(f"[QwenVL] PyTorch {torch.__version__}")
        print(f"[QwenVL] CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            current_fraction = torch.cuda.get_per_process_memory_fraction()
            print(f"[QwenVL] Current Memory Fraction: {current_fraction:.3f} ({current_fraction*100:.1f}% of GPU)")
            print(f"[QwenVL] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Allow user to set new fraction
            try:
                prompt = f"[QwenVL] Enter new memory fraction (0.1-0.9"
                if current_fraction is not None:
                    prompt += f", current={current_fraction:.3f})"
                new_fraction = float(input(prompt + ": ") or current_fraction)
                if 0.1 <= new_fraction <= 0.9:
                    torch.cuda.set_per_process_memory_fraction(new_fraction)
                    print(f"[QwenVL] ✅ Memory fraction set to: {new_fraction:.3f} ({new_fraction*100:.1f}%)")
                else:
                    print(f"[QwenVL] ❌ Invalid fraction. Must be between 0.1 and 0.9")
            except KeyboardInterrupt:
                print(f"[QwenVL] ✅ Keeping current fraction: {current_fraction:.3f}")
            except Exception as e:
                print(f"[QwenVL] ❌ Error setting fraction: {e}")
        else:
            print("[QwenVL] CUDA not available")
    except Exception as e:
        print(f"[QwenVL] Error checking PyTorch: {e}")

def set_pytorch_memory_fraction(fraction):
    """Set PyTorch memory fraction if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            if 0.1 <= fraction <= 0.9:
                torch.cuda.set_per_process_memory_fraction(fraction)
                print(f"[QwenVL] Memory fraction set to: {fraction:.3f} ({fraction*100:.1f}%)")
                return True
            else:
                print(f"[QwenVL] Invalid fraction: {fraction:.3f}. Must be between 0.1 and 0.9")
                return False
        else:
            print("[QwenVL] CUDA not available")
            return False
    except Exception as e:
        print(f"[QwenVL] Error: {e}")
        return False

def get_device_info():
    gpu = {"available": False, "total_memory": 0, "free_memory": 0}
    device_type = "cpu"
    recommended = "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = total - allocated - reserved
        
        gpu = {
            "available": True,
            "total_memory": total,
            "allocated_memory": allocated,
            "reserved_memory": reserved, 
            "free_memory": free,
        }
        device_type = "nvidia_gpu"
        recommended = "cuda"
        
        # Detailed memory debugging
        print(f"[QwenVL] GPU Memory Debug:")
        print(f"  Total VRAM: {total / 1024**3:.2f} GB")
        print(f"  Allocated: {allocated / 1024**3:.2f} GB")  
        print(f"  Reserved: {reserved / 1024**3:.2f} GB")
        print(f"  Free: {free / 1024**3:.2f} GB")
        print(f"  Model requires: 0.74 GB")
        print(f"  Available ratio: {(free / total) * 100:.1f}%")
        
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_type = "apple_silicon"
        recommended = "mps"
        gpu = {"available": True, "total_memory": 0, "free_memory": 0}
    sys_mem = psutil.virtual_memory()
    return {
        "gpu": gpu,
        "system_memory": {
            "total": sys_mem.total / 1024**3,
            "available": sys_mem.available / 1024**3,
        },
        "device_type": device_type,
        "recommended_device": recommended,
    }

def normalize_device_choice(device: str) -> str:
    device = (device or "auto").strip()
    if device == "auto":
        return "auto"

    if device.isdigit():
        device = f"cuda:{int(device)}"

    if device == "cuda":
        if not torch.cuda.is_available():
            print("[QwenVL] CUDA requested but not available, falling back to CPU")
            return "cpu"
        return "cuda"

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[QwenVL] CUDA requested but not available, falling back to CPU")
            return "cpu"
        if ":" in device:
            try:
                device_idx = int(device.split(":", 1)[1])
                if device_idx >= torch.cuda.device_count():
                    print(f"[QwenVL] CUDA device {device_idx} not available, using cuda:0")
                    return "cuda:0"
            except (ValueError, IndexError):
                print(f"[QwenVL] Invalid CUDA device format '{device}', using cuda:0")
                return "cuda:0"
        return device

    if device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            print("[QwenVL] MPS requested but not available, falling back to CPU")
            return "cpu"
        return "mps"

    return device

def flash_attn_available():
    if not torch.cuda.is_available():
        return False

    if platform.system().lower() != "linux":
        return False

    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False

    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False

    try:
        import importlib.metadata as importlib_metadata
        _ = importlib_metadata.version("flash_attn")
    except Exception:
        return False

    return True

def sage_attn_available():
    """Check if SageAttention is available and GPU supports it."""
    if not SAGE_ATTENTION_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major < 8:
        return False
    return True


def get_sage_attention_config():
    """Get the appropriate SageAttention kernel based on GPU architecture."""
    if not sage_attn_available():
        return None, None, None

    major, minor = torch.cuda.get_device_capability()
    arch_code = major * 10 + minor

    attn_func = None
    pv_accum_dtype = "fp32"

    if arch_code >= 120:  # Blackwell
        pv_accum_dtype = "fp32+fp32"
        attn_func = sageattn_qk_int8_pv_fp8_cuda
        print(f"[QwenVL] SageAttention: Using SM120 (Blackwell) FP8 kernel")
    elif arch_code >= 90:  # Hopper
        pv_accum_dtype = "fp32+fp32"
        attn_func = sageattn_qk_int8_pv_fp8_cuda_sm90
        print(f"[QwenVL] SageAttention: Using SM90 (Hopper) FP8 kernel")
    elif arch_code == 89:  # Ada Lovelace
        pv_accum_dtype = "fp32+fp32"
        attn_func = sageattn_qk_int8_pv_fp8_cuda
        print(f"[QwenVL] SageAttention: Using SM89 (Ada) FP8 kernel")
    elif arch_code >= 80:  # Ampere
        pv_accum_dtype = "fp32"
        attn_func = sageattn_qk_int8_pv_fp16_cuda
        print(f"[QwenVL] SageAttention: Using SM80+ (Ampere) FP16 kernel")
    else:
        print(f"[QwenVL] SageAttention not supported on SM{arch_code}")
        return None, None, None

    return attn_func, "per_warp", pv_accum_dtype

def is_fp8_model(model_name: str) -> bool:
    """Check if model name indicates it's a pre-quantized FP8 model."""
    fp8_indicators = ["-fp8", "_fp8", "-FP8", "_FP8"]
    return any(indicator in model_name for indicator in fp8_indicators)

def resolve_attention_mode(mode, force_sdpa=False):
    """Resolve attention mode with fallback logic.

    Args:
        mode: The requested attention mode
        force_sdpa: If True, always return SDPA (for FP8/BnB models)
    """
    if force_sdpa:
        return "sdpa"

    if mode == "sdpa":
        return "sdpa"
    if mode == "sage":
        if sage_attn_available():
            return "sage"
        print("[QwenVL] SageAttention forced but unavailable, falling back to SDPA")
        return "sdpa"
    if mode == "flash_attention_2":
        if flash_attn_available():
            return "flash_attention_2"
        print("[QwenVL] Flash-Attn forced but unavailable, falling back to SDPA")
        return "sdpa"

    # Auto mode: try sage → flash → sdpa
    if sage_attn_available():
        print("[QwenVL] Auto mode: Using SageAttention")
        return "sage"
    if flash_attn_available():
        print("[QwenVL] Auto mode: Using Flash Attention 2")
        return "flash_attention_2"
    print("[QwenVL] Auto mode: Using SDPA")
    return "sdpa"


def _model_snapshot_has_weights(target: Path) -> bool:
    return any(target.glob("*.safetensors")) or any(target.glob("*.bin"))


def _cleanup_unneeded_snapshot_weights(target: Path) -> None:
    unwanted_markers = ("fp32", "bf16", "fp8", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_0", "f32")
    for weight_file in target.rglob("*.safetensors"):
        name = weight_file.name.lower()
        if any(marker in name for marker in unwanted_markers):
            print(f"[QwenVL] Removing unneeded quantized weight: {weight_file.name}")
            weight_file.unlink(missing_ok=True)


def _model_snapshot_has_required_files(target: Path, require_processor: bool = False) -> bool:
    if not target.exists() or not target.is_dir():
        return False
    if not _model_snapshot_has_weights(target):
        return False
    if not (target / "config.json").exists():
        return False
    if require_processor:
        processor_files = (
            "preprocessor_config.json",
            "processor_config.json",
            "image_processor_config.json",
            "video_preprocessor_config.json",
        )
        if not any((target / name).exists() for name in processor_files):
            return False
    return True


def _download_model_snapshot(repo_id: str, target: Path, *, force_clean_target: bool = False, node_id=None, progress_label: str | None = None, hf_token: str | None = None) -> None:
    reporter = _DownloadProgressReporter(node_id=node_id, label=progress_label or "QwenVL HF Download", repo_id=repo_id)
    token_for_download = _clean_hf_token(hf_token)
    download_kwargs = {}
    if force_clean_target and target.exists() and target.is_dir():
        print(f"[QwenVL] Removing incomplete model snapshot before retry: {target}")
        reporter.stage("Cleaning incomplete snapshot", detail=target.name)
        shutil.rmtree(target, ignore_errors=False)
    reporter.stage("Preparing download", detail=target.name)
    try:
        download_kwargs = {
            "repo_id": repo_id,
            "local_dir": str(target),
            "ignore_patterns": ["*.md", ".git*"],
            "force_download": force_clean_target,
        }
        if token_for_download:
            download_kwargs["token"] = token_for_download
        tqdm_cls = reporter.make_tqdm_class()
        if tqdm_cls is not None:
            download_kwargs["tqdm_class"] = tqdm_cls
        snapshot_download(**download_kwargs)
        _cleanup_unneeded_snapshot_weights(target)
    except Exception as exc:
        reporter.fail(_hf_download_error_message(exc, repo_id=repo_id, hf_token=token_for_download))
        raise RuntimeError(_hf_download_error_message(exc, repo_id=repo_id, hf_token=token_for_download)) from exc
    finally:
        download_kwargs.pop("token", None)
        token_for_download = None
    reporter.finish(detail=target.name)


def download_hf_file_to_path(repo_ids: list[str], filename: str, target_path: Path, *, node_id=None, progress_label: str = "QwenVL GGUF Download", hf_token: str | None = None) -> None:
    if target_path.exists():
        print(f"[QwenVL] Using cached file: {target_path}")
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    wanted_name = Path(filename).name
    token_for_download = _clean_hf_token(hf_token)
    exact_candidates = []
    for candidate in (filename, wanted_name):
        if candidate and candidate not in exact_candidates:
            exact_candidates.append(candidate)

    try:
        for repo_id in repo_ids:
            reporter = _DownloadProgressReporter(node_id=node_id, label=progress_label, repo_id=repo_id)
            auth_label = "Preparing authenticated download" if token_for_download else "Preparing download"
            reporter.stage(auth_label, detail=wanted_name)
            if hf_hub_download is not None:
                for candidate in exact_candidates:
                    try:
                        reporter.stage("Resolving exact file", detail=candidate, force=False)
                        downloaded_path = _call_hf_hub_download(
                            repo_id=repo_id,
                            filename=candidate,
                            local_dir=target_path.parent,
                            token=token_for_download,
                            reporter=reporter,
                        )
                        if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            downloaded_path.replace(target_path)
                        if target_path.exists():
                            reporter.finish(detail=wanted_name)
                            return
                    except Exception as exc:
                        hint = _hf_download_error_message(exc, repo_id=repo_id, hf_token=token_for_download)
                        last_exc = RuntimeError(hint)
                        print(f"[QwenVL] exact file download failed from {repo_id}/{candidate}: {hint}")
            reporter.stage("Falling back to snapshot scan", detail=wanted_name, force=False)
            download_kwargs = {}
            try:
                download_kwargs = {
                    "repo_id": repo_id,
                    "repo_type": "model",
                    "local_dir": str(target_path.parent),
                    "allow_patterns": [filename, wanted_name, f"**/{wanted_name}"],
                }
                if token_for_download:
                    download_kwargs["token"] = token_for_download
                tqdm_cls = reporter.make_tqdm_class()
                if tqdm_cls is not None:
                    download_kwargs["tqdm_class"] = tqdm_cls
                snapshot_download(**download_kwargs)
                found = list(target_path.parent.rglob(wanted_name))
                if found:
                    downloaded_path = found[0]
                    if downloaded_path.exists() and downloaded_path.resolve() != target_path.resolve():
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        downloaded_path.replace(target_path)
                if target_path.exists():
                    reporter.finish(detail=wanted_name)
                    return
                reporter.fail(f"{wanted_name} was not found after snapshot download")
            except Exception as exc:
                hint = _hf_download_error_message(exc, repo_id=repo_id, hf_token=token_for_download)
                last_exc = RuntimeError(hint)
                reporter.fail(hint)
                print(f"[QwenVL] snapshot_download failed from {repo_id}: {hint}")
            finally:
                download_kwargs.pop("token", None)
    finally:
        token_for_download = None

    raise FileNotFoundError(f"[QwenVL] Download failed for {wanted_name}: {last_exc}")


def ensure_model(model_name, require_processor=False, node_id=None, progress_label: str | None = None, hf_token: str | None = None):
    info = HF_ALL_MODELS.get(model_name)
    if not info:
        raise ValueError(f"Model '{model_name}' not in config")

    # Local models have a direct path — use it, skip download
    local_path = info.get("local_path")
    if local_path:
        target = Path(local_path)
        if target.exists() and target.is_dir():
            return str(target)
        raise FileNotFoundError(f"[QwenVL] Local HF model directory not found: {target}")

    repo_id = info.get("repo_id")
    if not repo_id:
        raise ValueError(f"Model '{model_name}' has no repo_id or local_path")

    # Use ComfyUI's multi-path system if available
    llm_paths = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
    if llm_paths:
        models_dir = Path(llm_paths[0]) / "Qwen-VL"
    else:
        # Fallback to default behavior
        models_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"

    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / repo_id.split("/")[-1]

    # Only trust an existing snapshot if it also contains the metadata needed
    # by the selected backend. This avoids reusing stale or partial downloads.
    if _model_snapshot_has_required_files(target, require_processor=require_processor):
        return str(target)

    if target.exists() and target.is_dir():
        print(f"[QwenVL] Existing model snapshot is incomplete, refreshing: {target}")

    _download_model_snapshot(
        repo_id,
        target,
        node_id=node_id,
        progress_label=progress_label or f"QwenVL HF Download: {model_name}",
        hf_token=hf_token,
    )

    if not _model_snapshot_has_required_files(target, require_processor=require_processor):
        print(f"[QwenVL] Refreshed snapshot is still incomplete, retrying clean download: {target}")
        _download_model_snapshot(
            repo_id,
            target,
            force_clean_target=True,
            node_id=node_id,
            progress_label=progress_label or f"QwenVL HF Download: {model_name}",
            hf_token=hf_token,
        )

    if not _model_snapshot_has_required_files(target, require_processor=require_processor):
        missing = []
        if not (target / "config.json").exists():
            missing.append("config.json")
        if require_processor and not any(
            (target / name).exists()
            for name in (
                "preprocessor_config.json",
                "processor_config.json",
                "image_processor_config.json",
                "video_preprocessor_config.json",
            )
        ):
            missing.append("processor metadata")
        if not _model_snapshot_has_weights(target):
            missing.append("model weights")
        missing_str = ", ".join(missing) if missing else "required files"
        raise FileNotFoundError(
            f"[QwenVL] Downloaded model snapshot is incomplete for {repo_id}: missing {missing_str} in {target}"
        )

    return str(target)

def enforce_memory(model_name, quantization, device_info):
    info = HF_ALL_MODELS.get(model_name, {})
    requirements = info.get("vram_requirement", {})
    mapping = {
        Quantization.Q4: requirements.get("4bit", 0),
        Quantization.Q8: requirements.get("8bit", 0),
        Quantization.FP16: requirements.get("full", 0),
    }
    needed = mapping.get(quantization, 0)
    if not needed:
        return quantization  # Always return quantization
    
    if device_info["recommended_device"] in {"cpu", "mps"}:
        needed *= 1.5
        available = device_info["system_memory"]["available"]
    else:
        available = device_info["gpu"]["free_memory"]
    
    # More conservative memory management
    if needed * 1.5 > available:  # More conservative threshold
        if quantization == Quantization.FP16:
            print("[QwenVL] ⚠️  Auto-switch to 8-bit due to VRAM pressure")
            return Quantization.Q8
        if quantization == Quantization.Q8:
            print("[QwenVL] ⚠️  Auto-switch to 4-bit due to VRAM pressure")
            return Quantization.Q4
        raise RuntimeError(f"Insufficient memory for {quantization.value} mode. Required: {needed * 1.5:.1f}GB, Available: {available / 1024**3:.1f}GB")
    
    # Conservative memory check with safety margin
    if needed * 1.3 > available:  # Reduced from 1.5 to 1.3
        if quantization == Quantization.FP16:
            print("[QwenVL] ⚠️  Auto-switch to 8-bit due to VRAM pressure")
            return Quantization.Q8
        if quantization == Quantization.Q8:
            print("[QwenVL] ⚠️  Auto-switch to 4-bit due to VRAM pressure")
            return Quantization.Q4
        print(f"[QwenVL] Memory pressure detected. Consider: 1) Use 4-bit quantization, 2) Reduce max_tokens below 1024, 3) Close other applications")
        return quantization  # Always return quantization

def quantization_config(model_name, quantization):
    info = HF_ALL_MODELS.get(model_name, {})
    if info.get("quantized"):
        return None, None
    if quantization == Quantization.Q4:
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return cfg, None
    if quantization == Quantization.Q8:
        return BitsAndBytesConfig(load_in_8bit=True), None
    return None, torch.float16

class QwenVLBase:
    def __init__(self):
        self.device_info = get_device_info()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        print(f"[QwenVL] Node on {self.device_info['device_type']}")

    def clear(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def load_model(
        self,
        model_name,
        quant_value,
        attention_mode,
        use_compile,
        device_choice,
        keep_model_loaded,
        unique_id=None,
        hf_token: str | None = None,
    ):
        quant = Quantization.from_value(quant_value)  # Skip enforce_memory for now
        
        # Safety check: ensure quant is not None
        if quant is None:
            print(f"[QwenVL] Invalid quantization value: {quant_value}, falling back to FP16")
            quant = Quantization.FP16
        
        # Check if BitsAndBytes quantization is being used
        is_bnb_quantization = quant in [Quantization.Q4, Quantization.Q8]
        
        # Check if this is a pre-quantized FP8 model
        is_prequantized_fp8 = is_fp8_model(model_name) or HF_ALL_MODELS.get(model_name, {}).get("quantized", False)
        
        # Determine if we need to force SDPA (for FP8 or BitsAndBytes models)
        force_sdpa = is_prequantized_fp8 or is_bnb_quantization
        
        # Resolve attention mode with force_sdpa flag
        attn_impl = resolve_attention_mode(attention_mode, force_sdpa=force_sdpa)
        
        # Additional info messages for forced SDPA
        if force_sdpa and attention_mode in ["auto", "sage", "flash_attention_2"]:
            if is_prequantized_fp8:
                print("[QwenVL] FP8 model detected - forcing SDPA attention")
            elif is_bnb_quantization:
                print("[QwenVL] BitsAndBytes quantization detected - forcing SDPA attention")
        
        print(f"[QwenVL] Attention backend selected: {attn_impl}")
        
        device_requested = self.device_info["recommended_device"] if device_choice == "auto" else device_choice
        device = normalize_device_choice(device_requested)
        signature = (model_name, quant.value, attn_impl, device, use_compile)
        if keep_model_loaded and self.model is not None and self.current_signature == signature:
            ensure_cuda_vram_headroom("QwenVL", min_free_gb=1.0, min_free_ratio=0.08)
            return
        self.clear()
        model_path = ensure_model(
            model_name,
            require_processor=True,
            node_id=unique_id,
            progress_label=f"QwenVL HF Download: {model_name}",
            hf_token=hf_token,
        )
        quant_config, dtype = quantization_config(model_name, quant)
        
        # Handle attention mode for loading
        # SageAttention requires loading with SDPA first, then patching
        actual_attn_impl = attn_impl
        if attn_impl == "sage":
            actual_attn_impl = "sdpa"
        
        # MEMORY DEBUGGING: Check memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[QwenVL] 📊 Memory BEFORE load - Allocated: {allocated_before:.1f}GB, Reserved: {reserved_before:.1f}GB, Total: {total_memory:.1f}GB")
            
            # Check what's using memory
            if allocated_before > 2.0:
                print(f"[QwenVL] ⚠️  WARNING: {allocated_before:.1f}GB already allocated before loading!")
        
        # DEBUG: Print quantization config details
        print(f"[QwenVL] 🔍 DEBUG - Model: {model_name}, Quant: {quant}, Quant config: {quant_config}")
        print(f"[QwenVL] 🔍 DEBUG - Device: {device}, Dtype: {dtype}")
        print(f"[QwenVL] 🔍 DEBUG - Attention impl: {actual_attn_impl}")
        
        # Quantization is back - enforce_memory was the problem
        
        load_kwargs = {
            "device_map": "auto" if device != "cpu" and torch.cuda.is_available() else "cpu",
            "dtype": dtype or torch.float16,
            "attn_implementation": actual_attn_impl,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }

        if device != "cpu" and torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            gpu_mem_gb = max(1, int(gpu_mem / (1024**3)))
            load_kwargs["max_memory"] = {
                0: f"{gpu_mem_gb}GB",
                "cpu": "16GB",
            }
            
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
            
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
        
        # Apply SageAttention patching if needed
        if attn_impl == "sage":
            try:
                from sageattention_patch import set_sage_attention
                set_sage_attention(self.model)
                print("[QwenVL] SageAttention patching applied successfully")
            except Exception as e:
                print(f"[QwenVL] SageAttention patching failed: {e}")
                print("[QwenVL] Falling back to SDPA attention")
                # Model is already loaded with SDPA, so we can continue
        
        # MEMORY CLEANUP: Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[QwenVL] GPU memory after load: {torch.cuda.memory_allocated() / 1024**3:.1f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
                
        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
        if use_compile and device.startswith("cuda") and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[QwenVL] torch.compile enabled")
            except Exception as exc:
                print(f"[QwenVL] torch.compile skipped: {exc}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Detect architecture from config.json instead of relying on model name
        hf_model_type = read_hf_model_type(model_path)
        self.hf_model_type = hf_model_type
        self.is_qwen35 = hf_model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_5_vl") if hf_model_type else "qwen3.5-" in model_name.lower()
        self.supports_qwen_soft_think = hf_model_type == "qwen3" if hf_model_type else "qwen3-" in model_name.lower()
        if self.is_qwen35:
            print(f"[QwenVL] Qwen3.5 detected (model_type={hf_model_type}): Will disable thinking in chat template.")
        self.current_signature = signature

    @staticmethod
    def tensor_to_pil(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array, mode="L")
        if array.shape[-1] == 1:
            return Image.fromarray(array[..., 0], mode="L")
        if array.shape[-1] == 4:
            return Image.fromarray(array, mode="RGBA")
        return Image.fromarray(array[..., :3], mode="RGB")

    @torch.no_grad()
    def generate(
        self,
        prompt_text,
        image,
        video,
        frame_count,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        model_name="",
        stream_to_terminal=False,
        enable_thinking=True,
    ):
        # Memory optimization: clear cache before generation
        ensure_cuda_vram_headroom("QwenVL", min_free_gb=1.0, min_free_ratio=0.08)
        supports_soft_think = getattr(self, "supports_qwen_soft_think", False)
        is_qwen35 = getattr(self, "is_qwen35", False)
        context_window = resolve_qwen_context_window(getattr(self.model, "config", None))

        def _build_processed(thinking_enabled: bool):
            effective_prompt_text = apply_qwen_soft_thinking_directive(
                prompt_text,
                thinking_enabled,
                supports_soft_switch=supports_soft_think,
            )
            conversation = [{"role": "user", "content": []}]
            if image is not None:
                if image.dim() == 4 and image.shape[0] > 1:
                    print(f"[QwenVL] IMAGE input contains {image.shape[0]} items; using the first item only. Use the video input for multi-frame analysis.")
                conversation[0]["content"].append({"type": "image", "image": self.tensor_to_pil(image)})
            if video is not None:
                frames = [self.tensor_to_pil(frame) for frame in video]
                if len(frames) > frame_count:
                    idx = np.linspace(0, len(frames) - 1, frame_count, dtype=int)
                    frames = [frames[i] for i in idx]
                if frames:
                    conversation[0]["content"].append({"type": "video", "video": frames})
            conversation[0]["content"].append({"type": "text", "text": effective_prompt_text})

            chat_kwargs = {}
            if is_qwen35 or supports_soft_think:
                chat_kwargs["enable_thinking"] = thinking_enabled

            chat = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                **chat_kwargs
            )

            images = [item["image"] for item in conversation[0]["content"] if item["type"] == "image"]
            video_frames = [frame for item in conversation[0]["content"] if item["type"] == "video" for frame in item["video"]]
            videos = [video_frames] if video_frames else None
            processed = self.processor(text=chat, images=images or None, videos=videos, return_tensors="pt")
            return effective_prompt_text, processed

        requested_thinking = bool(enable_thinking)
        _, processed = _build_processed(requested_thinking)
        input_ids = processed.get("input_ids")
        prompt_tokens = int(input_ids.shape[-1]) if torch.is_tensor(input_ids) else 0
        effective_thinking = resolve_qwen_thinking_mode(
            enable_thinking,
            max_tokens,
            label="QwenVL HF",
            prompt_tokens=prompt_tokens,
            context_window=context_window,
        )
        if effective_thinking != requested_thinking:
            _, processed = _build_processed(effective_thinking)
        if supports_soft_think:
            directive = "/think" if effective_thinking else "/no_think"
            print(f"[QwenVL] Qwen3 detected: Thinking {'enabled' if effective_thinking else 'disabled'} via chat template and {directive}.")
        
        # Move to device more efficiently
        model_device = next(self.model.parameters()).device
        model_inputs = {
            key: value.to(model_device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }
        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)
        
        # Memory-efficient generation parameters
        kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if num_beams == 1:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
            # --- Qwen3.5 Heretic Logic: Top K ---
            if is_qwen35:
                kwargs["top_k"] = 20
                print("[QwenVL] Qwen3.5 detected: Forcing top_k=20 for recommended tuning.")
        else:
            kwargs["do_sample"] = False
            
        # Optional: staged readable terminal streaming
        if stream_to_terminal:
            try:
                from AILab_StreamDisplay import TerminalStreamDisplay
                from transformers import TextIteratorStreamer
                from threading import Thread
                stage_label = "STREAMING"
                stream_display = TerminalStreamDisplay("QwenVL HF", suppress_planning=True, compact=True)
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                kwargs["streamer"] = streamer
                thread = Thread(target=self.model.generate, kwargs={**model_inputs, **kwargs})
                print(f"[QwenVL HF] STREAMING {stage_label}")
                stream_display.start_stage(stage_label)
                stage_started_at = time.monotonic()
                last_status_at = stage_started_at
                thread.start()
                full_streamed = ""
                for token_str in streamer:
                    if token_str:
                        throw_exception_if_processing_interrupted()
                        full_streamed += token_str
                        stream_display.push_compact(token_str)
                        last_status_at = _maybe_emit_hf_stream_heartbeat(stage_label, stage_started_at, last_status_at, full_streamed)
                thread.join()
                stream_display.end_compact()
                stream_display.end_stage()
                if not full_streamed.strip():
                    raise RuntimeError("[QwenVL] HF streaming returned empty response")
                cleaned_text = full_streamed.strip()
                raw_text = full_streamed.strip()
                return cleaned_text, raw_text
            except ImportError:
                print("[QwenVL] TextIteratorStreamer not available — falling back to non-streaming")
                stream_to_terminal = False

        # Generate with interrupt support via background thread
        import threading, queue as qmod
        abort_event = threading.Event()
        output_queue: qmod.Queue = qmod.Queue()
        worker_error: Exception | None = None

        def _generate_worker():
            nonlocal worker_error
            try:
                result = self.model.generate(**model_inputs, **kwargs)
                output_queue.put(("ok", result))
            except Exception as exc:
                worker_error = exc
                output_queue.put(("error", None))

        worker = threading.Thread(target=_generate_worker, daemon=True)
        worker.start()
        try:
            while True:
                try:
                    kind, result = output_queue.get(timeout=0.25)
                except qmod.Empty:
                    throw_exception_if_processing_interrupted()
                    continue
                if kind == "error":
                    break
                outputs = result
                break
        finally:
            abort_event.set()
            worker.join(timeout=5.0)
        if worker_error:
            raise worker_error
        throw_exception_if_processing_interrupted()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        input_len = model_inputs["input_ids"].shape[-1]
        raw_text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()
        cleaned_text = raw_text  # generate() returns raw; caller cleans if needed
        return cleaned_text, raw_text

    def run(self, model_name, quantization, preset_prompt, custom_prompt, image, video, frame_count, max_tokens, temperature, top_p, num_beams, repetition_penalty, seed, keep_model_loaded, attention_mode, use_torch_compile, device, keep_last_prompt=False, unique_id=None, extra_pnginfo=None, node_class="QwenVL", stream_to_terminal=False, enable_thinking=True, hf_token: str | None = None):
        torch.manual_seed(seed)
        image_hash = get_image_hash(image)
        video_hash = get_video_hash(video)
        input_signature = build_node_input_signature(
            model_name=model_name,
            quantization=quantization,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            image_hash=image_hash,
            video_hash=video_hash,
            frame_count=frame_count,
            attention_mode=attention_mode,
            use_torch_compile=bool(use_torch_compile),
            device=device,
            enable_thinking=bool(enable_thinking),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        # Auto-retrieve saved prompt when seed is fixed
        saved = get_node_saved_prompt_with_seed(node_class, unique_id, extra_pnginfo, seed=seed, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, input_signature=input_signature)
        if saved and not stream_to_terminal:
            print(f"[QwenVL] Fixed seed {seed} matched — using per-node prompt: {saved[:50]}...")
            return (saved,)
        if saved and stream_to_terminal:
            print("[QwenVL] Streaming requested — bypassing fixed-seed prompt reuse for a fresh streamed run")
        if keep_last_prompt:
            print(f"[QwenVL] Keep last prompt enabled — looking up per-node state")
            if saved:
                print(f"[QwenVL] Using per-node prompt: {saved[:50]}...")
                return (saved,)
            else:
                print(f"[QwenVL] No per-node prompt found, returning empty")
                return ("",)
        
        # Always generate unless keep_last_prompt was requested
        print(f"[QwenVL] Generating new prompt")
        
        prompt_template = "" if preset_prompt == NO_PRESET_PROMPT else SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        
        # Generate cache key with all inputs including seed
        cache_key = get_cache_key(model_name, preset_prompt, custom_prompt, image_hash, video_hash, seed, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, enable_thinking=enable_thinking)
        
        # Check cache first (only for random mode)
        if cache_key in PROMPT_CACHE:
            cached_text = PROMPT_CACHE[cache_key].get("text", "")
            if cached_text:
                print(f"[QwenVL] Using cached prompt for seed {seed}: {cache_key[:8]}...")
                return (cached_text,)
        
        if custom_prompt and custom_prompt.strip():
            # Combine user input with template - custom prompt first for priority
            prompt = f"{custom_prompt.strip()}\n\n{prompt_template}" if prompt_template else custom_prompt.strip()
        else:
            prompt = prompt_template
            
        self.load_model(
            model_name,
            quantization,
            attention_mode,
            use_torch_compile,
            device,
            keep_model_loaded,
            unique_id=unique_id,
            hf_token=hf_token,
        )
        hf_token = None
        try:
            text, raw_text = self.generate(
                prompt,
                image,
                video,
                frame_count,
                max_tokens,
                temperature,
                top_p,
                num_beams,
                repetition_penalty,
                model_name=model_name,
                stream_to_terminal=stream_to_terminal,
                enable_thinking=enable_thinking,
            )
            
            # Cache the generated text
            PROMPT_CACHE[cache_key] = {
                "text": text,
                "timestamp": torch.cuda.Event().record() if torch.cuda.is_available() else None,
                "model": model_name,
                "preset": preset_prompt,
                "seed": seed,
                "image_hash": image_hash,
                "video_hash": video_hash
            }
            save_prompt_cache()  # Save cache to file
            
            print(f"[QwenVL] Cached new prompt for seed {seed}: {cache_key[:8]}...")
            
            # Save the generated prompt for future per-node keep-last-prompt
            set_node_saved_prompt(node_class, unique_id, extra_pnginfo, text, raw_trace=raw_text, seed=seed, max_tokens=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, input_signature=input_signature)
            print(f"[QwenVL] Saved per-node prompt: {text[:50]}...")
            
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()

class ThinkingLLM_QwenVL(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = list(HF_VL_MODELS.keys())
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        prompts = PRESET_PROMPTS or [NO_PRESET_PROMPT, "Describe this image in detail."]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]
        return {
            "required": {
                "model_name": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
            "preset_prompt": (prompts, {"default": default_prompt, "tooltip": TOOLTIPS["preset_prompt"] + "\n\nSelect 'No preset' to use only the custom prompt or image input."}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"]}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 8192, "tooltip": TOOLTIPS["max_tokens"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"] + "\n\n💡 Cache Info: Prompts are cached automatically. Use the same inputs (model, preset, custom prompt, image/video) to reuse cached prompts and avoid regeneration.\n\n🔒 Fixed Seed Mode: Set seed = 1 to ignore image/video changes and only use text-based caching. Perfect for keeping the same prompt regardless of media input variations."}),
                "keep_last_prompt": ("BOOLEAN", {"default": False, "tooltip": "Keep the last generated prompt instead of creating a new one"}),
                "stream_tokens_to_terminal": ("BOOLEAN", {"default": False, "tooltip": "Print every generated token live to the ComfyUI terminal/console"}),
                "enable_thinking": ("BOOLEAN", {"default": True, "tooltip": "Enable model reasoning/thinking when the backend supports it: True=allow thinking, False=force direct answer. Even when enabled, easy prompts may still get a direct answer, and this node automatically disables thinking when there is not enough output budget left for useful reasoning. For non-Qwen models (Gemma, LLaMA) this is advisory."}),
                "hf_token": ("STRING", {"default": "", "multiline": False, "tooltip": TOOLTIPS["hf_token"]}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "RAW_TRACE")
    FUNCTION = "process"
    CATEGORY = "ThinkingLLM"

    def process(self, model_name, preset_prompt, custom_prompt, attention_mode, max_tokens, keep_model_loaded, seed, keep_last_prompt=False, image=None, video=None, unique_id=None, extra_pnginfo=None, stream_tokens_to_terminal=False, enable_thinking=True, hf_token=""):
        # Always use FP16 - dropdown removed but keep working logic
        quantization = Quantization.FP16.value
        result = self.run(model_name, quantization, preset_prompt, custom_prompt, image, video, 16, max_tokens, 0.6, 0.9, 1, 1.2, seed, keep_model_loaded, attention_mode, False, "auto", keep_last_prompt, unique_id=unique_id, extra_pnginfo=extra_pnginfo, node_class="ThinkingLLM_QwenVL", stream_to_terminal=stream_tokens_to_terminal, enable_thinking=enable_thinking, hf_token=hf_token)
        hf_token = ""
        # Retrieve the raw_trace that was saved alongside the cleaned prompt in run()
        key = _make_node_state_key("ThinkingLLM_QwenVL", unique_id, extra_pnginfo)
        entry = NODE_PROMPT_STATE.get(key, {})
        raw_trace = entry.get("raw_trace", "") if isinstance(entry, dict) else ""
        return (result[0], raw_trace)

class ThinkingLLM_QwenVL_Advanced(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = list(HF_VL_MODELS.keys())
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        prompts = PRESET_PROMPTS or [NO_PRESET_PROMPT, "Describe this image in detail."]
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list

        return {
            "required": {
                "model_name": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "use_torch_compile": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["use_torch_compile"]}),
                "device": (device_options, {"default": "auto", "tooltip": TOOLTIPS["device"]}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": TOOLTIPS["preset_prompt"] + "\n\nSelect 'No preset' to use only the custom prompt or image input."}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"]}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 8192, "tooltip": TOOLTIPS["max_tokens"]}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "tooltip": TOOLTIPS["temperature"]}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "tooltip": TOOLTIPS["top_p"]}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": TOOLTIPS["num_beams"]}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "tooltip": TOOLTIPS["repetition_penalty"]}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "tooltip": TOOLTIPS["frame_count"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"] + "\n\n💡 Cache Info: Prompts are cached automatically. Use same inputs (model, preset, custom prompt, image/video) to reuse cached prompts and avoid regeneration.\n\n🔒 Fixed Seed Mode: Set seed = 1 to ignore image/video changes and only use text-based caching. Perfect for keeping the same prompt regardless of media input variations."}),
                "keep_last_prompt": ("BOOLEAN", {"default": False, "tooltip": "Keep last generated prompt instead of creating a new one"}),
                "stream_tokens_to_terminal": ("BOOLEAN", {"default": False, "tooltip": "Print every generated token live to the ComfyUI terminal/console"}),
                "enable_thinking": ("BOOLEAN", {"default": True, "tooltip": "Enable model reasoning/thinking when the backend supports it: True=allow thinking, False=force direct answer. Even when enabled, easy prompts may still get a direct answer, and this node automatically disables thinking when there is not enough output budget left for useful reasoning. For non-Qwen models (Gemma, LLaMA) this is advisory."}),
                "hf_token": ("STRING", {"default": "", "multiline": False, "tooltip": TOOLTIPS["hf_token"]}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "RAW_TRACE")
    FUNCTION = "process"
    CATEGORY = "ThinkingLLM"

    def process(self, model_name, attention_mode, use_torch_compile, device, preset_prompt, custom_prompt, max_tokens, temperature, top_p, num_beams, repetition_penalty, frame_count, keep_model_loaded, seed, keep_last_prompt, image=None, video=None, unique_id=None, extra_pnginfo=None, stream_tokens_to_terminal=False, enable_thinking=True, hf_token=""):
        # Always use FP16 - dropdown removed but keep working logic
        quantization = Quantization.FP16.value
        result = self.run(model_name, quantization, preset_prompt, custom_prompt, image, video, frame_count, max_tokens, temperature, top_p, num_beams, repetition_penalty, seed, keep_model_loaded, attention_mode, use_torch_compile, device, keep_last_prompt, unique_id=unique_id, extra_pnginfo=extra_pnginfo, node_class="ThinkingLLM_QwenVL_Advanced", stream_to_terminal=stream_tokens_to_terminal, enable_thinking=enable_thinking, hf_token=hf_token)
        hf_token = ""
        # Read back raw_trace from just-saved per-node state
        key = _make_node_state_key("ThinkingLLM_QwenVL_Advanced", unique_id, extra_pnginfo)
        entry = NODE_PROMPT_STATE.get(key, {})
        raw_trace = entry.get("raw_trace", "") if isinstance(entry, dict) else ""
        return (result[0], raw_trace)

NODE_CLASS_MAPPINGS = {
    "ThinkingLLM_QwenVL": ThinkingLLM_QwenVL,
    "ThinkingLLM_QwenVL_Advanced": ThinkingLLM_QwenVL_Advanced,
    "AILab_QwenVL": ThinkingLLM_QwenVL,
    "AILab_QwenVL_Advanced": ThinkingLLM_QwenVL_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThinkingLLM_QwenVL": "ThinkingLLM",
    "ThinkingLLM_QwenVL_Advanced": "ThinkingLLM Advanced",
    "AILab_QwenVL": "ThinkingLLM",
    "AILab_QwenVL_Advanced": "ThinkingLLM Advanced",
}
