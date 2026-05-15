# ComfyUI-ThinkingLLM

**ThinkingLLM** is a local-LLM custom-node pack for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that wraps Qwen3.5, Qwen3, and Qwen2.5-VL models behind a cleaner node interface with thinking-mode control, live token streaming, and a raw trace output for debugging.

📺 **[Watch the demo](https://youtu.be/K1JsvnzujOw)** for a better viewing experience.

> GPL-3.0 fork of Deaquay/ComfyUI-Qwen3.5-Uncensored, itself derived from huchukato/ComfyUI-QwenVL-Mod and 1038lab/ComfyUI-QwenVL. See [LICENSE](LICENSE).

![ThinkingLLM Demo](docs/DemoThinkingLLM%20(1).gif)
![Node screenshot](docs/Screenshot%20of%20Notes.png)

> 🎥 Prefer a narrated version? **[Watch the demo on YouTube](https://youtu.be/K1JsvnzujOw)**

## What ThinkingLLM adds

### The Thinking Toggle

Every ThinkingLLM node exposes an `enable_thinking` toggle.

- When it is on, the model may reason internally before answering. That can make a run take longer.
- For easy prompts, the model can still decide that explicit reasoning is unnecessary and answer directly. That is not a bug. It is a feature.
- When it is off, the node sends a `/no_think` style directive to discourage visible reasoning and push a more direct one-shot answer.

### Live token streaming

Enable `stream_tokens_to_terminal` to print tokens into the ComfyUI terminal as they arrive.

This is useful when:

- a prompt seems stuck and you want to see whether the model is actually working
- you want to debug a bad prompt while it is going wrong
- you want to understand whether the model is thinking or simply producing a long answer

### Raw trace output

Each node returns a second string output named `RAW_TRACE`.

- The main output is the cleaned answer for downstream workflow use.
- `RAW_TRACE` preserves the raw generation stream, including visible thinking blocks when the model emits them.

That makes it possible to inspect what the model thought about the prompt while still keeping the primary output clean.

## Nodes

### Transformers / HF nodes

This is the recommended default path.

- `ThinkingLLM`
- `ThinkingLLM (Advanced)`
- `ThinkingLLM Prompt Enhancer`

The HF path is the most straightforward option for Windows and Linux when you want the fewest install complications.

### GGUF / llama.cpp nodes

- `ThinkingLLM (GGUF)`
- `ThinkingLLM (GGUF Advanced)`
- `ThinkingLLM Prompt Enhancer (GGUF)`

The GGUF path requires a vision-capable `llama-cpp-python` build. The normal PyPI package is not enough for Qwen vision handlers. Use the setup notes in [docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md](docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md).

## Workflow tips

- **Pre-process input images** — use a resize or scale node before ThinkingLLM so large images don't blow up the context window.
- **Display the output** — connect RESPONSE or ENHANCED_OUTPUT to a **Show Text** or **Show Anything** node to see the generated text in the UI.
- **Inspect reasoning** — connect the RAW_TRACE output to a second Show Text node to see what the model was thinking.

## Supported models

ThinkingLLM supports these model families out of the box, with pre-configured entries in `hf_models.json` and `gguf_models.json`.

### Qwen — Vision-language (HF)
- Qwen3.5 4B / 9B
- Qwen3-VL 4B / 8B (Instruct, abliterated, unredacted)
- Qwen3-VL 32B Heretic
- Qwen2.5-VL 3B / 7B

### Qwen — Vision-language (GGUF)
- Qwen3.5 4B / 9B / 27B (Q4 K M through BF16)
- Qwen3-VL 4B / 8B / 32B (Instruct, abliterated, thinking)
- Qwen2.5-VL 3B / 7B

### Qwen — Text-only (HF)
- Qwen3 0.6B / 4B / 8B
- Qwen3.5 4B / 9B (base + heretic + uncensored)

### Qwen — Text-only (GGUF)
- Qwen3 4B / 8B (abliterated, Josiefied, base)
- Qwen3.5 4B / 9B / 27B (base + uncensored HauhauCS)

### Gemma 4 — Vision-language (HF)
- Gemma-4-E2B-it / Uncensored
- Gemma-4-E4B-it / Uncensored
- Gemma-4-26B-A4B-it / Heretic
- Gemma-4-31B-it

### Gemma 4 — Vision-language (GGUF)
- Gemma-4-E2B / E4B (Q4 K M through BF16, uncensored)
- Gemma-4-26B-A4B-it (Q4 K M through BF16)
- Gemma-4-31B-it (Q4 K M through BF16)

### Gemma 4 — Text-only (HF)
- Gemma-4-E2B-it / Uncensored
- Gemma-4-E4B-it / Uncensored
- Gemma-4-26B-A4B-it / Heretic
- Gemma-4-31B-it

Local GGUF models placed in `models/LLM/GGUF/` with a matching mmproj file are auto-discovered. Pre-configured entries live in `hf_models.json` and `gguf_models.json`.

## Quick install

### ComfyUI Manager (recommended)
1. Open **ComfyUI Manager** → **Install via Registry**
2. Search for `ThinkingLLM`
3. Click **Install**

### Manual install
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/goodguy1963/ComfyUI-ThinkingLLM.git
cd ComfyUI-ThinkingLLM
pip install -r requirements.txt
```

For GGUF vision support, follow [docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md](docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md).

## Model locations

Models are discovered automatically.

- HF models: `ComfyUI/models/LLM/Qwen-VL/`
- GGUF models: `ComfyUI/models/LLM/GGUF/`

For local GGUF models, keep the matching mmproj file beside the model file.

## Platform notes

- HF is the recommended default path and is the simplest cross-platform option.
- GGUF on Windows needs a matching `win_amd64` vision-capable wheel.
- GGUF on Linux needs the matching Linux wheel or compatible local build.
- Flash Attention support is best on Linux. The nodes fall back when it is unavailable.
- The thinking toggle works best with Qwen3.5 and Qwen3 style models. Other architectures may ignore the steering.

## FAQ

**Why does the node sometimes answer quickly and sometimes take longer?**

When thinking is enabled, the model may decide to reason before answering. For easy prompts it may skip explicit reasoning. That is expected behavior.

**How do I see what the model was thinking?**

Use the second `RAW_TRACE` output, or enable live terminal token streaming.

**How do I debug prompts that seem bad or stuck?**

Turn on `stream_tokens_to_terminal`. It is there specifically so you can see the generation live and catch bad prompt behavior sooner.

## Credits and fork lineage

This fork preserves the GPL-3.0 lineage of its predecessors.

- Deaquay/ComfyUI-Qwen3.5-Uncensored
- huchukato/ComfyUI-QwenVL-Mod
- 1038lab/ComfyUI-QwenVL
- Qwen Team (Alibaba Cloud)
- JamePeng/llama-cpp-python
- comfyanonymous/ComfyUI

Maintainer: goodguy1963  
Planned repo: [goodguy1963/ComfyUI-ThinkingLLM](https://github.com/goodguy1963/ComfyUI-ThinkingLLM)

## License

GPL-3.0. See [LICENSE](LICENSE).
