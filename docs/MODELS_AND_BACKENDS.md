# Models and backends

This guide describes ThinkingLLM's inference paths, curated model families, local discovery, and platform requirements.

## Choose a backend

### Transformers / Hugging Face

The HF path is the recommended default for Windows and Linux when you want the fewest installation complications. It powers:

- `ThinkingLLM`
- `ThinkingLLM (Advanced)`
- `ThinkingLLM Prompt Enhancer`

Some Gemma and Qwen tokenizers require `sentencepiece` or `tiktoken`. ThinkingLLM declares both dependencies and attempts to install a missing tokenizer backend at first load.

### GGUF / llama.cpp

The GGUF path powers:

- `ThinkingLLM (GGUF)`
- `ThinkingLLM (GGUF Advanced)`
- `ThinkingLLM Gemma 4 Audio (GGUF)`
- `ThinkingLLM Prompt Enhancer (GGUF)`

Multimodal GGUF inference requires a llama.cpp Python build with the chat handlers used by the selected architecture. The regular PyPI package may not include the required Qwen vision, Gemma 4, or MTMD audio support. Follow [Install llama-cpp-python](LLAMA_CPP_PYTHON_VISION_INSTALL.md) for platform-specific setup and verification.

### Whisper ASR

`ThinkingLLM Whisper ASR` uses `faster-whisper` for speech-to-text. It is independent of the Gemma 4 GGUF audio path and requires FFmpeg for general audio-file decoding.

### OpenAI-compatible API

The API node calls approved remote providers with credentials bound on the ComfyUI server. See [ThinkingLLM API](OPENAI_COMPATIBLE_API.md) for setup and security boundaries.

## Curated model families

ThinkingLLM ships model metadata in `hf_models.json` and `gguf_models.json`.

### Qwen vision-language

- **HF:** Qwen3.5 4B/9B; Qwen3-VL 4B/8B/32B; Qwen2.5-VL 3B/7B
- **GGUF:** Qwen3.8 27B; Qwen3.5 4B/9B/27B; Qwen3-VL 4B/8B/32B; Qwen2.5-VL 3B/7B

Catalog variants include selected Instruct, thinking, abliterated, heretic, unredacted, and uncensored releases.

### Qwen text-only

- **HF:** Qwen3 0.6B/4B/8B; Qwen3.5 4B/9B
- **GGUF:** Qwen3.8 27B; Qwen3 4B/8B; Qwen3.5 4B/9B/27B

### Gemma 4 vision-language

- **HF:** Gemma 4 E2B/E4B, 12B, 26B-A4B, and 31B families
- **GGUF:** Gemma 4 E2B/E4B, 12B, 26B-A4B, and 31B families across selected quantizations

### Gemma 4 text-only

Selected E2B, E4B, 12B, 26B-A4B, and 31B HF variants are available in text workflows.

### Gemma 4 audio

The audio selector includes curated audio-capable Gemma 4 E2B, E4B, and 12B GGUF entries. Audio also depends on the selected mmproj and a compatible `Gemma4ChatHandler`; a model name alone does not guarantee audio support.

### Whisper

Available sizes include `tiny`, `base`, `small`, `medium`, `large-v3`, and `distil-large-v3`.

The catalog files are the authoritative current list. Dropdown contents can evolve between releases as models and compatibility information change.

## Model locations

ThinkingLLM searches configured ComfyUI model paths and common defaults, including:

- HF snapshots: `ComfyUI/models/LLM/Qwen-VL/`
- GGUF models: `ComfyUI/models/LLM/GGUF/`
- compatible single-file checkpoints: `ComfyUI/models/text_encoders/`

Keep a GGUF model and its matching mmproj together. Local GGUF files with an mmproj are discovered even when they are not in the bundled catalog.

## Model labels

- `[installed]` means a catalog model was found in a configured `LLM`, `GGUF`, or `Qwen-VL` location.
- `[local]` means an uncatalogued compatible local model was discovered.

Labels are display-only. They do not alter saved model identities, and older saved display labels remain compatible where supported.

## Shared ComfyUI text encoders

Compatible checkpoints in `ComfyUI/models/text_encoders` appear in applicable selectors and are loaded through ComfyUI rather than copied into a second Transformers snapshot.

Supported examples include:

- Qwen3-VL 4B and 8B `.safetensors`, including `qwen3vl_8b_fp8_scaled.safetensors`
- Gemma 3 12B `.safetensors`, including `gemma-3-12b-it-heretic-v2_int8.safetensors`

The MiniMax H3/Qwen3-VL 32B conditioning checkpoint is excluded because it is not a complete text-generation model. ComfyUI-native generation supports one-beam sampling and prints terminal output after generation rather than streaming token-by-token.

## Community and commercial catalogs

Community dropdowns preserve normal model names and download behavior. Gated Hugging Face repositories still require acceptance of the model owner's terms and a read token.

Commercial-release dropdowns include only reviewed, locked models and group entries by rights status. This restriction is intentionally separate from Community behavior.

## Platform notes

- **Windows GGUF:** install a matching `win_amd64` multimodal wheel for the Python version bundled with ComfyUI.
- **Linux GGUF:** ThinkingLLM checks the backend at first use and can install a verified matching JamePeng wheel. Environment overrides are documented in the installation guide.
- **Gemma 4 audio:** requires multimodal `llama-cpp-python` 0.3.36 or newer and an mmproj that reports audio support.
- **Whisper:** requires `faster-whisper`; file-path decoding also requires FFmpeg.
- **Flash Attention:** support is strongest on Linux and falls back when unavailable.
- **Thinking controls:** Qwen3/Qwen3.5-style models generally respond most consistently to explicit thinking steering; other architectures may interpret it differently.

## Troubleshooting direction

- Missing or incompatible GGUF handlers: use [the llama.cpp installation guide](LLAMA_CPP_PYTHON_VISION_INSTALL.md).
- Unexpected response versus reasoning output: inspect `RAW_TRACE` and [Features and workflows](FEATURES_AND_WORKFLOWS.md).
- Remote provider setup or security errors: use [the API guide](OPENAI_COMPATIBLE_API.md).
- Duration or MiniMax structure questions: use [Duration-aware video prompts](VIDEO_PROMPT_DURATION.md).
