# Decision Router llama.cpp requirements

The ThinkingLLM Decision Router runs inference in the ComfyUI process. It does not launch `llama-server` or call an HTTP endpoint.

## Runtime capabilities

- **Text routing:** OpenJev 27B needs final-position logits; Clef-Flash 9B and Clef 27B need the native extended decision-batch API (`llama_batch_ext_*`, `llama_process`, and `llama_batch_ext_set_decision_order`). Both read native decision scores without generating an answer string.
- **Image routing:** OpenJev additionally needs MTMD image evaluation and Clef additionally needs MTMD image embedding APIs. The same native decision heads score the candidate routes after image encoding.

ThinkingLLM checks the selected model's required llama.cpp symbols before downloading model weights. Text routing does not require an MTMD handler or a vision projector; image routing checks for both. Missing capabilities produce a diagnostic in the ComfyUI error panel. The Image Router has been manually exercised in ComfyUI with Clef-Flash 9B; full-weight smoke tests for Clef 27B and OpenJev 27B, plus text-mode smoke tests, have not yet been recorded.

## Install a compatible build

Close ComfyUI before replacing llama-cpp-python. Use the exact Python interpreter that starts ComfyUI and select a JamePeng build whose compiled library exports the symbols above. A Python binding declaration alone is not sufficient: the symbols must exist in the native llama.cpp library too. The ThinkingLLM installer does not silently replace llama-cpp-python on Windows. A Windows x64 wheel for CPython 3.12 and CUDA 13.0 was verified with the router: JamePeng `0.4.5+cu130`, release `v0.4.5-cu130-win-20261025` (SHA-256 `188d68f915ad0ff10314ffd4e1d5ccb7376a3fd408ec24954143cfea0609abe1`). Other Python/CUDA combinations must be checked against their wheel's native symbols. The installer also does not replace an already importable backend just because a selected model's decision symbols are missing; do not try to upgrade the loaded native library from inside a running ComfyUI process.

On Linux, the installer's `THINKINGLLM_LLAMA_CPP_LINUX_SPEC` and `THINKINGLLM_LLAMA_CPP_LINUX_WHEEL_URL` settings are used only when the normal backend import fails and automatic installation is enabled. They do not trigger an upgrade when llama-cpp-python imports successfully but lacks the selected model's decision APIs. In that case, stop ComfyUI and install the selected compatible wheel or source build manually with the same Python interpreter, then restart. A source build remains opt-in with `THINKINGLLM_LLAMA_CPP_ALLOW_SOURCE_BUILD=1`; configure the CUDA toolchain first if GPU offload is required. These settings change only llama-cpp-python. They do not update PyTorch or Transformers.

After installing, restart ComfyUI so it loads the replacement native library. If a selected model still fails the capability check, the wheel or source build does not contain the required symbols. The node will stop before downloading the model.

## Models and licenses

Model GGUFs download on first use into `ComfyUI/models/LLM/GGUF`. The matching BF16 projector downloads only when an Image Router first uses that model:

| Model | Repository | Quantization | License |
| --- | --- | --- | --- |
| Clef-Flash 9B | [`ggml-org/Clef-Flash-GGUF`](https://huggingface.co/ggml-org/Clef-Flash-GGUF) | Q4_K_M | Apache-2.0 |
| Clef 27B | [`ggml-org/Clef-GGUF`](https://huggingface.co/ggml-org/Clef-GGUF) | Q4_K_M | Apache-2.0 |
| OpenJev 27B | [`ggml-org/OpenJev-GGUF`](https://huggingface.co/ggml-org/OpenJev-GGUF) | Q4_K_M | CC BY-NC-4.0 |

Commercial builds show the two Clef models only. OpenJev remains unavailable there because its model license is non-commercial. Commercial builds also follow ThinkingLLM's existing policy of requiring cleared models to be preinstalled rather than downloading weights at runtime.
