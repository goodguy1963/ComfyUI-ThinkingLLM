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

import os
import sys
from pathlib import Path

import torch

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from thinkingllm_core.model_access import (
    COMMERCIAL_RELEASE,
    MODEL_STATUS_PRESENTATION,
    _DownloadProgressReporter,
    _clean_hf_token,
    _download_model_snapshot,
    _hf_download_error_message,
    download_hf_file_to_path,
    enforce_model_access,
    hf_hub_download,
    model_catalog_label,
    model_catalog_options,
    normalize_commercial_status,
    resolve_model_catalog_name,
    snapshot_download,
)
from thinkingllm_core.media import (
    MASK_FOCUS_INSTRUCTION,
    apply_mask_highlight,
    get_image_hash,
    get_video_hash,
    tensor_to_pil,
)
from thinkingllm_core.prompt_contracts import (
    MINIMAX_H3_REFERENCE_PRESET,
    VIDEO_DURATION_INPUT,
    VIDEO_PRESET_METADATA,
    VIDEO_PROMPT_CONTRACT_VERSION,
    apply_qwen_soft_thinking_directive,
    apply_video_duration_context,
    ensure_video_prompt_duration,
    estimate_qwen_text_tokens,
    extract_minimax_source_dialogue,
    get_video_preset_metadata,
    normalize_minimax_prompt_best_effort,
    resolve_qwen_context_window,
    resolve_qwen_thinking_mode,
    resolve_video_duration,
    validate_minimax_reference_request,
    validate_minimax_source_duration,
)
from thinkingllm_core.prompt_state import (
    NODE_PROMPT_STATE,
    PROMPT_CACHE,
    _build_workflow_fingerprint,
    _make_node_state_key,
    build_node_input_signature,
    get_alternative_cache_key,
    get_cache_key,
    get_node_saved_prompt,
    get_node_saved_prompt_with_seed,
    load_node_prompt_state,
    load_prompt_cache,
    log_llm_input,
    save_node_prompt_state,
    save_prompt_cache,
    set_node_saved_prompt,
)

# Load cache and per-node state on module import
load_prompt_cache()
load_node_prompt_state()

# Export memory functions for external use
__all__ = [
    'PROMPT_CACHE', 'NODE_PROMPT_STATE',
    'get_cache_key', 'get_alternative_cache_key',
    'save_prompt_cache',
    'get_image_hash', 'get_video_hash', 'apply_mask_highlight', 'MASK_FOCUS_INSTRUCTION',
    'MINIMAX_H3_REFERENCE_PRESET', 'validate_minimax_reference_request',
    'VIDEO_PRESET_METADATA', 'VIDEO_DURATION_INPUT', 'VIDEO_PROMPT_CONTRACT_VERSION', 'get_video_preset_metadata',
    'resolve_video_duration', 'apply_video_duration_context', 'ensure_video_prompt_duration',
    'extract_minimax_source_dialogue', 'validate_minimax_source_duration',
    'normalize_minimax_prompt_best_effort',
    'check_pytorch_memory', 'set_pytorch_memory_fraction',
    'get_device_info', 'tensor_to_pil', 'enforce_memory',
    'quantization_config', 'ensure_model', 'resolve_attention_mode',
    'flash_attn_available', 'normalize_device_choice',
    'load_model_configs',
    'HF_VL_MODELS', 'HF_TEXT_MODELS', 'HF_ALL_MODELS',
    'model_catalog_label', 'model_catalog_options', 'resolve_model_catalog_name',
    'normalize_commercial_status', 'enforce_model_access',
    'SYSTEM_PROMPTS', 'PRESET_PROMPTS', 'TOOLTIPS',
    'Quantization', 'ATTENTION_MODES',
    'NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS',
    'get_node_saved_prompt_with_seed', 'get_node_saved_prompt', 'set_node_saved_prompt',
    'load_node_prompt_state', 'save_node_prompt_state',
    '_make_node_state_key',
]
import folder_paths

from thinkingllm_core.hf_models import (
    ATTENTION_MODES,
    HF_ALL_MODELS,
    HF_TEXT_MODELS,
    HF_VL_MODELS,
    NO_PRESET_PROMPT,
    PRESET_PROMPTS,
    SYSTEM_PROMPTS,
    TOOLTIPS,
    Quantization,
    _default_model_from_config,
    check_pytorch_memory,
    enforce_memory,
    ensure_cuda_vram_headroom,
    ensure_model,
    flash_attn_available,
    get_device_info,
    get_sage_attention_config,
    is_fp8_model,
    load_model_configs,
    normalize_device_choice,
    quantization_config,
    read_hf_model_type,
    resolve_attention_mode,
    sage_attn_available,
    set_pytorch_memory_fraction,
)
from thinkingllm_core.hf_runtime import QwenVLBase

class ThinkingLLM_QwenVL(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = model_catalog_options(HF_VL_MODELS)
        default_model = _default_model_from_config(HF_VL_MODELS, "Qwen3-VL-4B-Instruct")
        default_model = model_catalog_label(default_model, HF_VL_MODELS.get(default_model))
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
                "mask": ("MASK",),
                "duration_seconds": VIDEO_DURATION_INPUT,
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

    def process(self, model_name, preset_prompt, custom_prompt, attention_mode, max_tokens, keep_model_loaded, seed, keep_last_prompt=False, image=None, video=None, mask=None, duration_seconds=5.0, unique_id=None, extra_pnginfo=None, stream_tokens_to_terminal=False, enable_thinking=True, hf_token=""):
        # Always use FP16 - dropdown removed but keep working logic
        quantization = Quantization.FP16.value
        result = self.run(model_name, quantization, preset_prompt, custom_prompt, image, video, 16, max_tokens, 0.6, 0.9, 1, 1.2, seed, keep_model_loaded, attention_mode, False, "auto", keep_last_prompt, unique_id=unique_id, extra_pnginfo=extra_pnginfo, node_class="ThinkingLLM_QwenVL", stream_to_terminal=stream_tokens_to_terminal, enable_thinking=enable_thinking, hf_token=hf_token, mask=mask, duration_seconds=duration_seconds)
        hf_token = ""
        # Retrieve the raw_trace that was saved alongside the cleaned prompt in run()
        key = _make_node_state_key("ThinkingLLM_QwenVL", unique_id, extra_pnginfo)
        entry = NODE_PROMPT_STATE.get(key, {})
        raw_trace = entry.get("raw_trace", "") if isinstance(entry, dict) else ""
        return (result[0], raw_trace)

class ThinkingLLM_QwenVL_Advanced(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = model_catalog_options(HF_VL_MODELS)
        default_model = _default_model_from_config(HF_VL_MODELS, "Qwen3-VL-4B-Instruct")
        default_model = model_catalog_label(default_model, HF_VL_MODELS.get(default_model))
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
                "mask": ("MASK",),
                "duration_seconds": VIDEO_DURATION_INPUT,
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("RESPONSE", "RAW_TRACE", "MASK_PREVIEW")
    FUNCTION = "process"
    CATEGORY = "ThinkingLLM"

    def process(self, model_name, attention_mode, use_torch_compile, device, preset_prompt, custom_prompt, max_tokens, temperature, top_p, num_beams, repetition_penalty, frame_count, keep_model_loaded, seed, keep_last_prompt, image=None, video=None, mask=None, duration_seconds=5.0, unique_id=None, extra_pnginfo=None, stream_tokens_to_terminal=False, enable_thinking=True, hf_token=""):
        # Always use FP16 - dropdown removed but keep working logic
        quantization = Quantization.FP16.value
        result = self.run(model_name, quantization, preset_prompt, custom_prompt, image, video, frame_count, max_tokens, temperature, top_p, num_beams, repetition_penalty, seed, keep_model_loaded, attention_mode, use_torch_compile, device, keep_last_prompt, unique_id=unique_id, extra_pnginfo=extra_pnginfo, node_class="ThinkingLLM_QwenVL_Advanced", stream_to_terminal=stream_tokens_to_terminal, enable_thinking=enable_thinking, hf_token=hf_token, mask=mask, duration_seconds=duration_seconds)
        hf_token = ""
        # Read back raw_trace from just-saved per-node state
        key = _make_node_state_key("ThinkingLLM_QwenVL_Advanced", unique_id, extra_pnginfo)
        entry = NODE_PROMPT_STATE.get(key, {})
        raw_trace = entry.get("raw_trace", "") if isinstance(entry, dict) else ""
        mask_preview, _ = apply_mask_highlight(image, mask)
        return (result[0], raw_trace, mask_preview)

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
