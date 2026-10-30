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
- The public GitHub repository is `https://github.com/goodguy1963/ComfyUI-ThinkingLLM`; Registry publication runs automatically when `pyproject.toml` changes on `main`.
- Registry and ComfyUI-Manager publication use the semantic version in `pyproject.toml` as the source of truth.
- Keep the same version mirrored in the `__init__.py` metadata block and `custom_nodes.json` so local ComfyUI metadata and manager-facing metadata stay aligned.
- README still contains historical upstream links and changelog references beyond the top-level fork identity; that is acceptable for attribution, but a docs pass should normalize publish-facing instructions before release.

## ThinkingLLM 3.1.0 release record

Version metadata is synchronized at **3.1.0** in `pyproject.toml`, `__init__.py`, and `custom_nodes.json`. The release adds typed Boolean and bounded Integer Decision Fields for image and text inputs, local and API inference, dynamic native-type output sockets, and a readable `decision` summary alongside raw `decision_json`. The feature idea was contributed by [JMarimon1](https://www.reddit.com/user/JMarimon1/); see `update.md` for detailed changes and verification.

## ThinkingLLM 3.0.0 release record

Version metadata is synchronized at **3.0.0** in `pyproject.toml`, `__init__.py`, and `custom_nodes.json`. The detailed user-facing changes and validation status are recorded in `update.md`.

### Changes that need to ship together

- Include the Decision Router node, its native runtime adapter, dynamic route UI, commercial frontend variant, tests, and user-facing documentation.
- Require in-process `llama-cpp-python` support for llama.cpp SystemOne decision batches and MTMD vision. The router must not depend on a separate `llama-server`.
- Keep the runtime capability check before model downloads and keep actionable diagnostics for missing decision-head, vision, or CUDA backend APIs.
- Preserve the dynamic CUDA backend probe fix in `AILab_LlamaCppInstaller.py`; newer wheels need their backend DLLs registered before `llama_supports_gpu_offload()` reports GPU support.
- Do not update PyTorch or Transformers as a side effect of setting up the llama.cpp runtime.
- Keep model weights and projectors outside the repository. Download them on first use into ComfyUI's `models/LLM/GGUF` directory. Keep OpenJev out of commercial builds because its model license is non-commercial.

### llama.cpp compatibility and install path

- The `pyproject.toml` dependency remains unpinned (`llama-cpp-python`). A clean install can resolve to a build without the native SystemOne or MTMD APIs. The runtime guide documents the manual compatibility check and one verified Windows combination; capability detection alone does not install a compatible Windows runtime.
- One local Windows x64 / CPython 3.12 / CUDA 13.0 combination was verified with the [JamePeng 0.4.5+cu130 release wheel](https://github.com/JamePeng/llama-cpp-python/releases/tag/v0.4.5-cu130-win-20261025), SHA-256 `188d68f915ad0ff10314ffd4e1d5ccb7376a3fd408ec24954143cfea0609abe1`. It exposes Clef decision batches, MTMD vision, and dynamic CUDA backends on the tested machine. Treat this as one verified configuration, not a portable version pin for all users.
- This local runtime was installed with `--no-deps`; PyTorch and Transformers were left unchanged. `pip check` reports existing environment conflicts, including NumPy 2.3.5 against the llama-cpp-python metadata cap of 2.3.2. The previous 0.3.36 package had the same cap. Do not blanket-upgrade the ComfyUI environment to resolve this unrelated metadata mismatch.

### Release validation and remaining coverage

- Version metadata is synchronized, and the README, install guide, workflow guide, commercial frontend, and release log document the Decision Router.
- The user manually exercised the Image Router in ComfyUI with Clef-Flash 9B on fox, dog, cat, and none routes and observed the intended outputs and branch behavior.
- All 220 automated tests pass, and the Image Router has been manually exercised with Clef-Flash 9B. Clean install validation and full-weight inference for Clef 27B and OpenJev 27B remain open coverage items; see `update.md` for measured checks.
- Only Windows x64 / CPython 3.12 / CUDA 13.0 with the documented JamePeng wheel is locally verified. Other OS, Python, and CUDA combinations remain unverified.
