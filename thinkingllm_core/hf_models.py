"""Hugging Face model configuration, discovery, device, and attention support."""

import gc
import json
import os
import platform
from enum import Enum
from pathlib import Path

import psutil
import torch
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

import folder_paths

from thinkingllm_core.model_access import (
    COMMERCIAL_RELEASE,
    _download_model_snapshot,
    _model_snapshot_has_required_files,
    enforce_model_access,
    normalize_commercial_status,
    resolve_model_catalog_name,
)
from thinkingllm_core.prompt_contracts import VIDEO_PRESET_METADATA

try:
    from sageattention.core import (
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp8_cuda,
        sageattn_qk_int8_pv_fp8_cuda_sm90,
    )
    SAGE_ATTENTION_AVAILABLE = True
except Exception:
    sageattn_qk_int8_pv_fp16_cuda = None
    sageattn_qk_int8_pv_fp8_cuda = None
    sageattn_qk_int8_pv_fp8_cuda_sm90 = None
    SAGE_ATTENTION_AVAILABLE = False

NODE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = NODE_DIR / ("hf_models.commercial.json" if COMMERCIAL_RELEASE else "hf_models.json")
SYSTEM_PROMPTS_PATH = NODE_DIR / "AILab_System_Prompts.json"
HF_VL_MODELS: dict[str, dict] = {}
HF_TEXT_MODELS: dict[str, dict] = {}
HF_ALL_MODELS: dict[str, dict] = {}
SYSTEM_PROMPTS = {}
NO_PRESET_PROMPT = "🚫 No preset (image-only)"
PRESET_PROMPTS: list[str] = [NO_PRESET_PROMPT, "Describe this image in detail."]


TOOLTIPS = {
    "model_name": "Pick the checkpoint. [ComfyUI] entries reuse compatible models/text_encoders files; other entries download into models/LLM/Qwen-VL on first use.",
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
    next_vl_models: dict[str, dict] = {}
    next_text_models: dict[str, dict] = {}
    next_system_prompts: dict = {}
    next_preset_prompts = list(PRESET_PROMPTS)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        if "hf_vl_models" in data or "hf_text_models" in data:
            next_vl_models = data.get("hf_vl_models") or {}
            next_text_models = data.get("hf_text_models") or {}
        else:
            next_vl_models = {k: v for k, v in data.items() if not k.startswith("_")}
        if COMMERCIAL_RELEASE:
            next_vl_models = {name: info for name, info in next_vl_models.items() if normalize_commercial_status(info) == "cleared"}
            next_text_models = {name: info for name, info in next_text_models.items() if normalize_commercial_status(info) == "cleared"}
        next_system_prompts = data.get("_system_prompts", {})
        next_preset_prompts = data.get("_preset_prompts", next_preset_prompts)
    except Exception as exc:
        print(f"[QwenVL] Config load failed: {exc}")
    try:
        with open(SYSTEM_PROMPTS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh) or {}
        qwenvl_prompts = data.get("qwenvl") or {}
        preset_override = data.get("_preset_prompts") or []
        video_metadata = data.get("_video_presets") or {}
        if isinstance(qwenvl_prompts, dict) and qwenvl_prompts:
            next_system_prompts = qwenvl_prompts
        if isinstance(preset_override, list) and preset_override:
            next_preset_prompts = preset_override
        if isinstance(video_metadata, dict):
            VIDEO_PRESET_METADATA.clear()
            VIDEO_PRESET_METADATA.update(
                {
                    name: dict(entry)
                    for name, entry in video_metadata.items()
                    if isinstance(name, str) and isinstance(entry, dict)
                }
            )
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[QwenVL] System prompts load failed: {exc}")

    if NO_PRESET_PROMPT not in next_preset_prompts:
        next_preset_prompts = [NO_PRESET_PROMPT, *next_preset_prompts]
    if isinstance(next_system_prompts, dict):
        next_system_prompts.setdefault(NO_PRESET_PROMPT, "")
    custom = NODE_DIR / "custom_models.json"
    if custom.exists() and not COMMERCIAL_RELEASE:
        try:
            with open(custom, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            custom_vl = data.get("hf_vl_models") or {}
            custom_text = data.get("hf_text_models") or {}
            legacy = data.get("hf_models", {}) or data.get("models", {})
            if isinstance(custom_vl, dict) and custom_vl:
                next_vl_models.update(custom_vl)
                print(f"[QwenVL] Loaded {len(custom_vl)} custom VL models")
            if isinstance(custom_text, dict) and custom_text:
                next_text_models.update(custom_text)
                print(f"[QwenVL] Loaded {len(custom_text)} custom text models")
            if isinstance(legacy, dict) and legacy:
                next_vl_models.update(legacy)
                print(f"[QwenVL] Loaded {len(legacy)} custom legacy models")
        except Exception as exc:
            print(f"[QwenVL] custom_models.json skipped: {exc}")

    HF_VL_MODELS.clear()
    HF_VL_MODELS.update(next_vl_models)
    HF_TEXT_MODELS.clear()
    HF_TEXT_MODELS.update(next_text_models)
    HF_ALL_MODELS.clear()
    HF_ALL_MODELS.update(HF_VL_MODELS)
    HF_ALL_MODELS.update(HF_TEXT_MODELS)
    SYSTEM_PROMPTS.clear()
    SYSTEM_PROMPTS.update(next_system_prompts)
    PRESET_PROMPTS[:] = next_preset_prompts

    # Commercial releases expose only the two entries in the reviewed catalog.
    if not COMMERCIAL_RELEASE:
        _scan_local_hf_models()


def _scan_local_hf_models():
    """Scan local HF directories and compatible ComfyUI text encoders."""
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
                    "commercial_status": "local",
                    "quantized": False,
                }
                HF_VL_MODELS[display] = model_info
                HF_ALL_MODELS[display] = model_info
                known_dirs.add(entry.name)
                count += 1
        except PermissionError:
            pass

    try:
        text_encoders = folder_paths.get_filename_list("text_encoders")
    except (AttributeError, KeyError):
        text_encoders = []
    for filename in text_encoders:
        normalized = Path(filename).name.lower().replace("-", "").replace("_", "")
        family = None
        if "gemma312b" in normalized:
            family = "gemma3"
        elif "qwen3vl" in normalized and ("4b" in normalized or "8b" in normalized):
            family = "qwen3vl"
        if not family or not normalized.endswith(".safetensors"):
            continue
        display = f"[ComfyUI] {filename}"
        model_info = {
            "comfy_text_encoder": filename,
            "repo_id": None,
            "is_local": True,
            "commercial_status": "local",
            "quantized": True,
            "native_family": family,
        }
        HF_VL_MODELS[display] = model_info
        HF_ALL_MODELS[display] = model_info
        count += 1

    if count:
        print(f"[QwenVL] Discovered {count} local model(s) on disk")


if not HF_ALL_MODELS:
    load_model_configs()


def _default_model_from_config(models: dict[str, dict], fallback: str) -> str:
    for name, info in models.items():
        if isinstance(info, dict) and info.get("default"):
            return name
    return next(iter(models.keys()), fallback)


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



def ensure_model(model_name, require_processor=False, node_id=None, progress_label: str | None = None, hf_token: str | None = None):
    model_name = resolve_model_catalog_name(HF_ALL_MODELS, model_name)
    info = HF_ALL_MODELS.get(model_name)
    if not info:
        raise ValueError(f"Model '{model_name}' not in config")

    # Local models have a direct path — use it, skip download
    local_path = info.get("local_path")
    if local_path:
        target = Path(local_path)
        if target.exists() and target.is_dir():
            enforce_model_access(info, model_name, local_exists=True, hf_token=hf_token)
            return str(target)
        enforce_model_access(info, model_name, local_exists=False, hf_token=hf_token)
        raise FileNotFoundError(f"[QwenVL] Local HF model directory not found: {target}")

    repo_id = info.get("repo_id")
    if not repo_id:
        raise ValueError(f"Model '{model_name}' has no repo_id or local_path")

    if COMMERCIAL_RELEASE:
        enforce_model_access(info, model_name, local_exists=False, hf_token=hf_token)
        target = Path(folder_paths.models_dir) / "LLM" / repo_id.split("/")[-1]
        if _model_snapshot_has_required_files(target, require_processor=require_processor):
            enforce_model_access(info, model_name, local_exists=True, hf_token=hf_token)
            return str(target)
        raise FileNotFoundError(
            f"[QwenVL] Locked commercial model is missing or incomplete: {target}"
        )

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
        enforce_model_access(info, model_name, local_exists=True, hf_token=hf_token)
        return str(target)

    enforce_model_access(info, repo_id, local_exists=False, hf_token=hf_token)

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
