# ComfyUI-ThinkingLLM Maintainer Notes

## Fork lineage

- License: GPL-3.0 is preserved via the inherited [LICENSE](../LICENSE).
- Primary fork lineage: Deaquay/ComfyUI-Qwen3.5-Uncensored.
- Prior lineage referenced in code/docs: huchukato/ComfyUI-QwenVL-Mod and 1038lab/ComfyUI-QwenVL.
- Internal `Qwen` and `QwenVL` identifiers remain in code where they are technically meaningful and lower-risk to keep.

## Platform support from the current codebase

- HF nodes are cross-platform in principle because they use `transformers`, `torch`, `huggingface-hub`, `Pillow`, `opencv-python`, and optional `bitsandbytes`/`accelerate`.
- HF attention auto-detection is now Linux-only for `flash_attn`; non-Linux platforms fall back to SDPA unless SageAttention is available.
- GGUF nodes require a vision-capable `llama-cpp-python` build that exposes `Qwen3VLChatHandler` or `Qwen25VLChatHandler`.
- Windows GGUF setup is wheel-driven in [LLAMA_CPP_PYTHON_VISION_INSTALL.md](./LLAMA_CPP_PYTHON_VISION_INSTALL.md) and should use a matching `win_amd64` wheel for the exact Python and CUDA runtime.
- Linux GGUF setup can use the matching Linux CUDA wheel from the same guide; optional Linux-only `flash_attn` wheels are documented in the existing README.

## Publishability notes

- Metadata has been rebranded to `ComfyUI-ThinkingLLM` with publisher id `goodguy1963`.
- Repository URLs currently point at the planned GitHub location `https://github.com/goodguy1963/ComfyUI-ThinkingLLM` and need a real remote repo before registry publication.
- README still contains historical upstream links and changelog references beyond the top-level fork identity; that is acceptable for attribution, but a docs pass should normalize publish-facing instructions before release.
