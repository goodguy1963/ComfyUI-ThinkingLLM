# ThinkingLM GGUF (llama.cpp) Performance Issues — Developer Notes

**Version**: ThinkingLLM v1.0.8 (ComfyUI custom node)
**User**: Running on a high-end Linux server that should outperform a Windows home PC but doesn't
**Date**: May 2026

---

## 0. Current Review and Implementation Status

**Overall rating**: 8.5/10 as a developer forensic note; 7/10 as a user-facing ComfyUI troubleshooting note.

This note earns a good rose: it is unusually strong for a ComfyUI maintainer note because it connects user-visible slowness to backend installation, GPU-offload verification, dropped llama.cpp kwargs, and server CPU-thread behavior. Compared with many ComfyUI node notes, it is more diagnostic and actionable, but also heavier than a normal user guide.

The main weakness is freshness. Several sections below describe the pre-remediation state and should now be read as historical context unless the status table says otherwise. Keep this file as the maintainer forensic note; keep `docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md` as the user-facing quick path.

### Implementation Coverage

| Original item | Current status | Notes |
| ------------- | -------------- | ----- |
| Detect CUDA beyond `torch.version.cuda` | Implemented | Backend diagnostics now report multiple CUDA evidence sources instead of relying only on torch. |
| Avoid silent Linux source-build fallback | Implemented | Unverified Linux source builds now require explicit opt-in or override. |
| Verify backend GPU-offload support | Implemented | Backend probing and checker script cover this path. Real L40S validation is still pending. |
| Add backend diagnostic line | Implemented | Backend status is available in terminal and RAW_TRACE-style diagnostics. |
| Surface dropped llama.cpp performance kwargs | Implemented | Dropped kwargs and warnings are included in GGUF backend trace output. |
| Warn about thread defaults on large servers | Implemented | Thread guidance is surfaced through backend warnings and recommended actions. |
| Add diagnostic script | Implemented | `tools/check_llama_backend.py` exists and can run a model-load plus tiny inference smoke test when a local GGUF path is supplied. |
| Update user docs | Implemented | `docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md` covers install, tokens, and slow CPU-fallback troubleshooting. |
| Add Hugging Face token handling | Implemented beyond original note | Nodes include `hf_token`, environment-token fallback, token redaction, exact-file download, and frontend clearing after execution. |
| Clean up GGUF Advanced UI | Implemented beyond original note | Frontend hides stable internal widgets while preserving widget order compatibility. |
| Separate `enable_thinking` from model reload signature | Implemented | Qwen template-thinking state is synced onto the live `Llama` instance before generation, so toggling it no longer forces reload by signature. |
| Add Windows auto-install | Deferred | Keep manual wheel install unless a safe Windows wheel mapping and DLL-path strategy are validated. |
| Real L40S benchmark | Pending external validation | Requires target Linux/L40S machine and real model files. |

### Important Correction

The CUDA 12.4 vs 12.9 mismatch should be treated as evidence to report, not a definitive root cause by itself. Current code should prefer evidence-based diagnostics: torch CUDA, toolkit CUDA, driver CUDA, backend package version, vision handler availability, and GPU-offload probing.

### Remaining Plan

1. Run a real target-machine backend smoke test:

    ```bash
    python tools/check_llama_backend.py --model <path-to-gguf> --gpu-layers 1 --verbose-load --strict-gpu
    ```

2. Run a real L40S before/after benchmark once the reporter can provide access to the target machine and model files.

3. Decide separately whether Windows auto-install is worth implementing. Keep it deferred unless wheel mapping, Python-version matching, and DLL loading can be validated safely.

4. Add measured tokens/sec output to `tools/check_llama_backend.py` only if it can stay safe for arbitrary local GGUF files and avoid misleading results on tiny prompts.

## 1. Environment Summary

### The Server That Runs Slowly (this machine)

| Component | Detail |
| --------- | ------ |
| GPU | **NVIDIA L40S** — 46 GB VRAM, Ada Lovelace (SM 8.9), 350W TDP |
| CUDA driver | 560.35.03 — system CUDA version **12.9** |
| PyTorch CUDA | **12.4** (mismatch with system CUDA 12.9!) |
| OS | Linux |
| Python | 3.12 (from `/venv/main/lib/python3.12/`) |
| ComfyUI | Launched via `start_comfyui.py` with `--use-sage-attention` |
| Node in use | ThinkingLLM (GGUF) / ThinkingLLM (GGUF Advanced) |

### The Windows Home Machine That Runs Faster

- Windows ComfyUI portable install
- Weaker consumer GPU
- Manually installed `llama-cpp-python` with CUDA vision wheel

---

## 2. Root Cause Summary

The performance regression on the Linux server is almost certainly caused by **llama-cpp-python running without proper GPU acceleration**, combined with several passive code paths that silently degrade LLM inference to CPU-only without notifying the user. The plugin's Linux auto-installer gives false confidence while delivering sub-optimal or CPU-only backend builds.

---

## 3. Critical Issues

### 3.1 ⚠️ CRITICAL: Linux Wheel Auto-Mapping is Too Narrow and Silently Falls Back to Source Build

**File**: `AILab_LlamaCppInstaller.py`, lines 9–45 and 194–235

The known Linux wheel table (`KNOWN_LINUX_WHEEL_SPECS`) only covers **three CUDA versions**:

| CUDA | Python | Status |
| ---- | ------ | ------ |
| 12.4 | cp312 | Has wheel |
| 12.8 | cp310–cp314 | Has wheels |
| 13.0 | cp312 | Has wheel |

**This server's system CUDA is 12.9** (not in the table), and PyTorch reports **CUDA 12.4**. The auto-installer uses PyTorch's reported CUDA version (from `torch.version.cuda`) to pick a wheel. In this case it would pick the CUDA 12.4 wheel — but CUDA 12.4 and system CUDA 12.9 is a **major/minor mismatch** (12.4 vs 12.9). The CUDA 12.4 wheel is compiled against CUDA 12.4 libraries; loading it against a CUDA 12.9 runtime is unpredictable:

- If the cu124 wheel silently loads but fails to initialize CUDA kernels → **CPU-only inference**
- If the cu124 wheel links against symbols that changed between 12.4 and 12.9 → **crash or missing GPU functions**
- Either way, the model falls back to CPU, giving the illusion of working but running at 1/10th the speed

For any CUDA version not in the table (12.5, 12.6, 12.7, 12.9, 12.10, etc.), the fallback is:

```python
DEFAULT_JAMEPENG_GIT_SPEC = "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
```

This triggers a **source build** of llama-cpp-python, which:

- Is compiled without CUDA unless the build environment has the CUDA toolkit installed (not guaranteed in a venv)
- Takes many minutes to build — the user thinks it "installed successfully" but may have zero CUDA support
- Does not validate CUDA capability after building

**Why this explains Linux being slower than Windows**:

- On Windows, the plugin has **NO auto-installer** (line 198: `if platform.system().lower() != "linux" … raise RuntimeError`). Windows users MUST manually download and install a JamePeng CUDA wheel. This manual step ensures they get the exact wheel matching their Python + CUDA + platform.
- On Linux, the auto-installer picks the wrong wheel (or builds from source without CUDA), and the user gets CPU-only inference while thinking it "just works."

### 3.2 ⚠️ CRITICAL: No Verification That llama-cpp-python Actually Uses GPU After Install

**File**: `AILab_LlamaCppInstaller.py`, line 228–235 (`_import_llama_cpp_backend`)

After installing llama-cpp-python, the plugin only checks that `Llama` can be imported and that vision handlers exist. It never verifies that:

1. CUDA kernels are actually callable in the installed build
2. `n_gpu_layers` can be ≥ 1 (i.e., GPU offloading works)
3. A simple test inference completes on GPU

A failing auto-install gives this error:

```text
[QwenVL] llama_cpp was installed, but the backend is still incompatible.
```

But a "successful" install (where `Llama` imports fine but has no CUDA) gives **no warning at all**. The model loads with `n_gpu_layers=-1` but llama-cpp-python silently ignores it and runs entirely on CPU.

### 3.3 ⚠️ HIGH: `_filter_kwargs_for_callable` Silently Drops All Performance Parameters

**File**: `AILab_QwenVL_GGUF.py`, lines 438–452 and 888–903

When passing kwargs to the llama-cpp-python `Llama()` constructor, the code uses `_filter_kwargs_for_callable()` to remove any parameter that the installed `Llama.__init__` doesn't accept:

```python
llm_kwargs_filtered = _filter_kwargs_for_callable(getattr(Llama, "__init__", Llama), llm_kwargs)
dropped_perf_kwargs = [
    key for key in ("n_ubatch", "n_threads", "n_threads_batch", 
                     "flash_attn", "offload_kqv", "ctx_checkpoints")
    if key in llm_kwargs and key not in llm_kwargs_filtered
]
if dropped_perf_kwargs:
    print("[QwenVL] Warning: installed llama_cpp Llama() does not accept performance kwargs: ...")
```

This is a good safety mechanism, but the problem is:

- The warning is **only printed to the console** — the ComfyUI UI never shows it
- If `flash_attn` is dropped, llama.cpp runs without flash attention (slower)
- If `offload_kqv` is dropped, KV cache stays on CPU (dramatically slower)
- If `n_ubatch` is dropped, batch processing uses llama.cpp defaults (potentially sub-optimal)
- The user has no way to know these optimizations were silently disabled

This is especially harmful combined with Issue 3.1: if the auto-installed build is a stripped-down version (e.g., without CUDA flash attention support), ALL performance parameters get dropped, and the model runs in the slowest possible configuration.

### 3.4 ⚠️ HIGH: Default `n_threads=None` / `n_threads_batch=None` on High-Core-Count Servers

**File**: `AILab_QwenVL_GGUF.py`, lines 768–769 and `ThinkingLLM_QwenVL_GGUF` node (line 1442) and `Advanced` node (lines 1542–1543)

Both the basic and advanced GGUF nodes default to `n_threads=None` and `n_threads_batch=None`, which means llama.cpp auto-detects the thread count. On a server-class machine with the L40S (typically paired with 32–128 CPU cores), llama.cpp's auto-detection can use too many threads, causing:

- **NUMA memory access penalties**: threads on different sockets fight for shared memory
- **CPU cache thrashing**: too many threads competing for L2/L3 cache
- **Memory bandwidth saturation**: consumer GPUs expect fewer concurrent memory streams
- **Token generation slowdown**: the prompt processing phase (which uses `n_threads_batch`) can actually be slower with too many threads

A Windows consumer PC with 8–16 cores would avoid all of these problems naturally through fewer auto-detected threads.

The `n_threads` and `n_threads_batch` controls exist in the Advanced node UI (hidden behind an Advanced tab), but the defaults never set optimal values. A typical sweet spot for llama.cpp on 32+ core servers is `n_threads=8` to `n_threads=16` and `n_threads_batch` = physical core count (not logical).

### 3.5 ⚠️ HIGH: `_detect_cuda_version()` Reports PyTorch's CUDA, Not System CUDA

**File**: `AILab_LlamaCppInstaller.py`, lines 67–73

```python
def _detect_cuda_version() -> str:
    import torch
    return str(getattr(torch.version, "cuda", "") or "")
```

This returns the CUDA version PyTorch was **compiled against**, NOT the actual CUDA runtime version on the system. On this server:

- System CUDA: **12.9** (`nvidia-smi` reports 12.9)
- PyTorch CUDA: **12.4** (`torch.version.cuda`)

The auto-installer therefore picks a CUDA **12.4** wheel for a system running CUDA **12.9**. This is architecturally wrong:

| What the cu124 wheel expects | What this server has |
| ---------------------------- | -------------------- |
| libcublas.so.12 (CUDA 12.4 API) | libcublas.so.12 (CUDA 12.9 API) |
| libcudart.so.12 (CUDA 12.4 API) | libcudart.so.12 (CUDA 12.9 API) |

The SONAME is the same (`libcublas.so.12`) so the dynamic linker won't reject it, but internal API/ABI differences between CUDA 12.4 and 12.9 can cause silent failures where kernels run on CPU fallback paths inside llama.cpp without any error message.

### 3.6 ⚠️ MEDIUM: `flash_attn_available()` Is Linux-Only for HF Path, Creating Asymmetric Behavior

**File**: `AILab_QwenVL.py`, lines 963–969

```python
def flash_attn_available():
    if platform.system().lower() != "linux":
        return False
```

This is **good and correct** for the HuggingFace model path (Flash Attention 2 doesn't work on Windows). However, it creates an asymmetry in the overall plugin behavior that makes debugging GGUF issues harder:

- On Linux HF path: SageAttention > Flash Attention 2 > SDPA (all 3 available, fast)
- On Linux GGUF path: depends entirely on llama-cpp-python build (potentially CPU-only)
- On Windows HF path: SageAttention > SDPA (Flash Attention 2 disabled)
- On Windows GGUF path: depends on manually installed llama-cpp-python wheel

A user who tests the HF node on Linux and sees fast performance might assume the GGUF node is equally fast, not realizing they depend on completely different backends.

### 3.7 ⚠️ MEDIUM: GGUF Model Reloads on Every Minor Parameter Change

**File**: `AILab_QwenVL_GGUF.py`, lines 781–801

The model signature check includes `bool(enable_thinking)` and `ctx_checkpoints_val` among other parameters:

```python
signature = (
    str(model_path),
    str(mmproj_path) if has_mmproj else "",
    n_ctx, n_gpu_layers, n_batch_val, n_ubatch_val, device_kind,
    img_max, top_k_val, pool_size_val, n_threads_val, n_threads_batch_val,
    bool(flash_attn), bool(offload_kqv), ctx_checkpoints_val,
    bool(enable_thinking),  # <-- changes trigger full reload!
)
```

Every time `enable_thinking` toggles, the entire model is unloaded and reloaded. On a server loading a 8B-parameter GGUF model from disk/NFS, this can add 5–30 seconds per toggle. Combined with `keep_model_loaded=False` (default), every workflow run pays the full load cost.

### 3.8 LOW: No Diagnostic Logging About Which Backend Build Was Actually Installed

There is no output like:

```text
[QwenVL] llama-cpp-python v0.3.34+cu124.basic installed — CUDA: YES, Vision: YES
```

The only way to know is to manually inspect the installed package in the venv. This makes remote debugging nearly impossible.

---

## 4. Hardware-Specific Factors on This Server

### 4.1 NVIDIA L40S (Ada Lovelace, SM 8.9)

| Property | Value |
| -------- | ----- |
| VRAM | 46,068 MB |
| FP16 TFLOPS | ~91 (with sparsity) |
| Memory bandwidth | 864 GB/s |
| CUDA compute capability | 8.9 |
| SageAttention support | FP8 kernel (`sageattn_qk_int8_pv_fp8_cuda`) |

This GPU should absolutely crush inference — it's faster than an RTX 4090 in pure compute. If llama-cpp-python is running on CPU, this hardware advantage is completely wasted.

### 4.2 CUDA Version Mismatch (system 12.9 vs torch 12.4)

```text
nvidia-smi → Driver 560.35.03, CUDA 12.9
torch.version.cuda → "12.4"
```

The 560.35.03 driver is fully compatible with CUDA 12.9. The PyTorch compiled for CUDA 12.4 works fine because CUDA is backward compatible. However, the **llama-cpp-python auto-installer** using the wrong CUDA version to select a wheel is the problem — it should probe the **system** CUDA version (via `nvidia-smi` or `nvcc --version`), not just PyTorch's.

### 4.3 Potential NUMA Configuration

Server-class machines often have multi-socket CPUs with NUMA memory regions. llama.cpp can suffer significantly from sub-optimal thread pinning. If `n_threads` auto-detects 64+ cores on a dual-socket server, threads on different NUMA nodes compete for memory bandwidth, slowing inference 2-3x compared to pinning threads to a single NUMA node.

---

## 5. The Windows vs Linux Performance Paradox

Here's why the Windows home PC can outperform this Linux server:

| Factor | Linux Server | Windows Home PC |
| ------ | ------------ | --------------- |
| llama-cpp-python install | Auto-installer picks wrong/no wheel → CPU fallback | Manual install of exact CUDA wheel → GPU accelerated |
| GPU layers | `n_gpu_layers=-1` silently ignored on CPU-only build | `n_gpu_layers=-1` works, all layers on GPU |
| Flash attention (llama.cpp) | `flash_attn=True` silently dropped if not in API | `flash_attn=True` works on CUDA build |
| KV cache offload | `offload_kqv=True` silently dropped | `offload_kqv=True` works, KV cache on GPU |
| Thread count | Auto-detects 32–128 cores → potential thrashing | Auto-detects 8–16 cores → efficient |
| Verification | None after auto-install | User verified `Qwen3VLChatHandler` import manually |

---

## 6. Recommended Fixes (For the Developer)

### 6.1 Critical Fixes

1. **Add system CUDA detection**: Don't just use `torch.version.cuda`. Also probe the system via `nvidia-smi` or by checking `CUDA_VERSION` in the environment or `/usr/local/cuda/version.txt`. Use the **system** CUDA version to select the correct JamePeng wheel.

2. **Verify CUDA capability after auto-install**: After installing llama-cpp-python, run a minimal smoke test:

     ```python
     from llama_cpp import Llama
     # Verify CUDA is available in the build
     import ctypes
     # Check if ggml CUDA backend loads
     ```

3. **Add a `[QwenVL] Backend:` diagnostic line** at startup that prints:

     ```text
     [QwenVL] llama-cpp-python v0.3.34+cu124.basic — CUDA: YES, Vision: YES, FA: YES
     ```

4. **Expand known wheel versions** in `KNOWN_LINUX_WHEEL_SPECS` to cover more CUDA versions (12.5, 12.6, 12.7, 12.9) and provide clear error messages when no match is found, rather than silently falling back to a source build.

### 6.2 High-Priority Fixes

1. **Add `n_threads` warnings**: When `n_threads` is left at None on a system with >16 physical cores, print a warning suggesting the user set it manually.

2. **Make dropped performance parameters visible in the UI**: When `_filter_kwargs_for_callable` drops `flash_attn`, `offload_kqv`, etc., surface this as a node status or a field in the RAW_TRACE output.

3. **Add a `CUDA backend check` button or auto-verify**: Before running inference, verify that the llama-cpp-python build actually supports GPU layers by doing a quick `Llama()` with `n_gpu_layers=1` and checking that it doesn't report CPU-only.

### 6.3 Medium-Priority Fixes

1. **Provide Windows auto-install**: Currently there is NO auto-install on Windows (`if platform.system().lower() != "linux" ... raise RuntimeError`). The irony is that Windows users get a more reliable experience because they're forced to follow the manual install guide. Add Windows auto-install with the same care.

2. **Add a diagnostic test script**: Ship a `check_backend.py` script that tests:

   - llama-cpp-python import
   - Vision handler availability
   - CUDA kernel availability
   - Quick inference benchmark

3. **Separate `enable_thinking` from model signature**: Don't include `enable_thinking` in the model reload signature — it can be changed without reloading the entire model since it's just a chat template parameter.

### 6.4 Documentation

1. **Update `docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md`** with explicit CUDA version matching instructions and a troubleshooting section for the "runs but slow" scenario (CPU fallback).

---

## 7. What the User Should Check Right Now

While waiting for a fix, the user should run these diagnostics on the Linux server:

```bash
# 1. Check what version of llama-cpp-python is installed and whether it has CUDA
python -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null || echo "NOT INSTALLED"

# 2. Check if vision handlers are present
python -c "from llama_cpp.llama_chat_format import Qwen3VLChatHandler; print('Vision: OK')" 2>/dev/null || echo "Vision: MISSING"

# 3. Check if llama-cpp-python was built with CUDA
python -c "import llama_cpp; import ctypes; lib = llama_cpp._lib; print('Has ggml_cuda:', hasattr(lib, 'ggml_cuda_has_device') or 'ggml_cuda' in str(lib.__dict__))" 2>/dev/null || echo "CUDA check failed"

# 4. List all installed llama_cpp wheels
pip show llama-cpp-python 2>/dev/null || echo "Not installed via pip"
```

Then manually install the correct wheel for CUDA 12.9 + Python 3.12 + Linux x64 from JamePeng's releases if available, or set:

```bash
THINKINGLLM_LLAMA_CPP_LINUX_WHEEL_URL=<correct-wheel-url>
```

---

## 8. Summary

The ThinkingLLM GGUF node's Linux performance regression is a three-way problem:

1. **The auto-installer picks the wrong build** (CUDA version mismatch, falls back to source build without CUDA)
2. **No verification** that the installed backend actually uses GPU acceleration
3. **Performance parameters silently dropped** without user-visible warning

The plugin works correctly — the models run, tokens are generated — but the GPU sits idle while the CPU does all the work. On a machine with an L40S GPU, this is the difference between generating 100 tokens/second on GPU and generating 2–3 tokens/second on CPU.

The reason it's faster on a Windows home PC is that Windows users are forced to manually install llama-cpp-python, and in doing so, they pick the correct CUDA wheel for their exact setup. The Linux auto-installer, intended to make things easier, ends up making things worse by installing a non-GPU build silently.

---

**Written for**: goodguy1963 (ThinkingLLM maintainer)
**Repository**: [goodguy1963/ComfyUI-ThinkingLLM](https://github.com/goodguy1963/ComfyUI-ThinkingLLM)
**Issue contact**: User on Linux server with L40S, CUDA 12.9, PyTorch cu124, Python 3.12
