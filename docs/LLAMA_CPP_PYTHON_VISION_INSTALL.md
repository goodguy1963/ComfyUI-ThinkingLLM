# Install `llama-cpp-python` (Vision / Qwen-VL GGUF)

This plugin’s **QwenVL (GGUF)** vision nodes require a `llama-cpp-python` build that includes multimodal chat handlers such as:

- `Qwen3VLChatHandler`
- `Qwen25VLChatHandler`
- `Gemma4ChatHandler` (`llama-cpp-python` 0.3.36 or newer for Gemma 4 audio)

The upstream `llama-cpp-python` from PyPI often does **not** include these vision handlers. Use a fork/build that provides them (e.g. JamePeng’s fork) and install a **Release wheel**.

Release wheels (download `.whl` here):

- [https://github.com/JamePeng/llama-cpp-python/releases](https://github.com/JamePeng/llama-cpp-python/releases/)

## 0) Close ComfyUI first

Stop ComfyUI before installing/replacing packages, especially on Windows portable.

## Linux auto-install in ThinkingLLM

On Linux, ThinkingLLM now performs a backend check the first time you run a GGUF node.

- If `llama_cpp` is missing or lacks the required Qwen vision handlers, it attempts an automatic install.
- The first automatic choice is a known JamePeng Linux wheel when the runtime matches a built-in mapping.
- If there is no verified built-in wheel match, it now stops with a clear diagnostic instead of silently building a possibly CPU-only backend.
- Source builds are opt-in. When enabled and a CUDA toolkit is detected, ThinkingLLM sets `CMAKE_ARGS=-DGGML_CUDA=on` and `FORCE_CMAKE=1` for the install.

Environment overrides:

- `THINKINGLLM_AUTO_INSTALL_LLAMA_CPP=0` disables automatic Linux installation.
- `THINKINGLLM_LLAMA_CPP_LINUX_WHEEL_URL=<wheel-url>` forces a specific Linux wheel.
- `THINKINGLLM_LLAMA_CPP_LINUX_SPEC=<pip-spec>` forces an explicit pip install target.
- `THINKINGLLM_LLAMA_CPP_ALLOW_SOURCE_BUILD=1` allows the JamePeng Git source-build fallback when no verified wheel is known.
- `THINKINGLLM_LLAMA_CPP_ALLOW_CPU_SOURCE_BUILD=1` allows a source build even when no CUDA toolkit is detected. Use this only when you intentionally want CPU inference.
- `THINKINGLLM_LLAMA_CPP_VERBOSE_LOAD=1` enables verbose llama.cpp model-load logs so you can inspect `offloaded X/Y layers`, `CUDA0 buffer`, KV cache placement, and `flash_attn` status.

After an automatic install, restart ComfyUI if the current process does not pick up the new backend cleanly.

## Hugging Face token for gated/private downloads

ThinkingLLM nodes now include an optional `hf_token` field for Hugging Face downloads. Use it when a model, GGUF file, or mmproj file is private or gated.

- The token is passed only to the Hugging Face download call.
- The token is not added to cache keys, RAW_TRACE, backend diagnostics, or fixed-seed prompt signatures.
- Download errors redact the token before printing messages.
- The local in-memory token variable is cleared after each download attempt.
- If `hf_token` is empty, downloads can use `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` from the ComfyUI process environment.
- Clear the `hf_token` field before saving or sharing workflows, because ComfyUI workflows can serialize widget values.

If a download reports `401`, `403`, `Unauthorized`, `Forbidden`, `Repository Not Found`, or `gated`, check both of these:

- The Hugging Face account behind the token has accepted the model license/gate.
- The token has read permission for the model repository.

If you do not need private or gated models, leave `hf_token` empty.

### CUDA version notes

ThinkingLLM reports CUDA evidence separately:

- **Torch CUDA**: the CUDA version PyTorch was compiled for, such as `12.4`.
- **Toolkit CUDA**: the CUDA toolkit found through `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda/version.txt`, or `nvcc`.
- **nvidia-smi CUDA**: the maximum CUDA version advertised by the installed NVIDIA driver.

A minor-version difference inside CUDA 12.x is not, by itself, proof that llama.cpp is CPU-only. The reliable check is whether the installed `llama-cpp-python` library supports GPU offload and whether verbose llama.cpp load logs show GPU buffers/layers.

Run the standalone probe from this folder:

```bash
python tools/check_llama_backend.py
```

For a real model-load smoke test:

```bash
python tools/check_llama_backend.py --model /path/to/model.gguf --gpu-layers 1 --verbose-load --strict-gpu
```

## 1) Identify the exact Python ComfyUI uses

### Windows portable (common)

Your Python is usually:

`ComfyUI\\python_embeded\\python.exe`

Check:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -V
C:\AI\ComfyUI\python_embeded\python.exe -c "import sys; print(sys.executable)"
```

### venv / conda

Activate your env, then:

```bash
python -V
python -c "import sys; print(sys.executable)"
```

## 2) Backup your environment (recommended)

```bat
C:\AI\ComfyUI\python_embeded\python.exe -m pip freeze > C:\AI\ComfyUI\requirements-backup.txt
```

## 3) Install the Release wheel (recommended)

Download a **Release `.whl`** from:
[https://github.com/JamePeng/llama-cpp-python/releases/](https://github.com/JamePeng/llama-cpp-python/releases/)

The wheel **must match ALL of the following**:

- **Python version** used by ComfyUI
  (`cp310` / `cp311` / `cp312` / `cp313`)
- **Platform**
  `win_amd64` (Windows 64-bit)
- **Build type**

  - **CPU wheel** → safest option (no CUDA toolkit required)
  - **CUDA wheel (`cuXXX`)** → requires a compatible CUDA runtime / toolkit

> [!WARNING]
> **Windows note (important)**
> If you install a CUDA wheel, the CUDA build tag (e.g. `cu121`, `cu122`) must be compatible with your installed CUDA runtime/toolkit.

A mismatch can cause errors like **“cannot load ggml.dll” even though the file exists**.

If you are unsure, **use a CPU wheel**.

Install with force-reinstall (safer than manual uninstall):

```bat
C:\AI\ComfyUI\python_embeded\python.exe -m pip install --upgrade --force-reinstall C:\path\to\llama_cpp_python-*.whl
```

Notes:

- Warnings about leftover folders like `~umpy` are usually safe to ignore while ComfyUI is closed.
- Make sure ComfyUI is **fully stopped** before installing.

## 4) Verify vision handlers exist

```bat
C:\AI\ComfyUI\python_embeded\python.exe -c "from llama_cpp.llama_chat_format import Qwen3VLChatHandler, Qwen25VLChatHandler; print('handlers OK')"
```

If this fails, you installed a wheel that does not include vision support (or installed into the wrong Python environment).

## 4b) Verify the backend is not silently CPU-only

The GGUF node now prints a line like:

```text
[QwenVL] llama.cpp backend: llama-cpp-python 0.3.x (cp312, x86_64); GPU offload=yes; vision=Qwen3VLChatHandler; CUDA torch=12.4, toolkit=12.8, nvidia-smi=12.9; module=...
```

In the Advanced node, RAW_TRACE begins with a `[BACKEND]` section. Check these fields first when a strong GPU runs slowly:

- `GPU offload=yes/no/unknown`
- `dropped_performance_kwargs`
- `warnings`
- `recommended_actions`
- `device`, `gpu_layers`, `flash_attn`, and `offload_kqv`

If GPU offload is `no`, install a CUDA-capable wheel or rebuild with CUDA. If it is `unknown`, set `THINKINGLLM_LLAMA_CPP_VERBOSE_LOAD=1`, restart ComfyUI, and look for llama.cpp load lines such as `offloaded X/Y layers`, `CUDA0 buffer size`, `CUDA0 KV buffer size`, and `flash_attn = 1`.

On high-core Linux servers, leaving `n_threads=0` and `n_threads_batch=0` can be slower than a smaller fixed value because llama.cpp may span NUMA nodes. Start with `n_threads=8` to `16`, then tune `n_threads_batch` for prompt processing.

## 5) Fix common dependency conflicts (Windows)

Some wheels may upgrade dependencies (notably `numpy` / `pillow`) and cause conflicts with other packages (like OpenCV).

### OpenCV conflict (recommended fix)

If you see errors like:

- `opencv-python ... requires numpy<2.3.0,>=2; ... but you have numpy 2.3.x`

Pin numpy back:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -m pip install --upgrade "numpy<2.3"
```

### Pillow conflicts (optional)

If you don’t use packages that depend on an older Pillow, you can ignore Pillow warnings. Otherwise:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -m pip install --upgrade "pillow<12"
```
