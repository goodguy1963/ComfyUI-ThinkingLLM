import contextlib
import importlib
import importlib.metadata
import os
import platform
import re
import subprocess
import sys

DEFAULT_JAMEPENG_GIT_SPEC = "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
KNOWN_LINUX_WHEEL_SPECS = {
    ("cp312", "linux_x86_64", "12.4"): (
        "https://github.com/JamePeng/llama-cpp-python/releases/download/"
        "v0.3.34-cu124-Basic-linux-20260331/"
        "llama_cpp_python-0.3.34+cu124.basic-cp312-cp312-linux_x86_64.whl"
    ),
    ("cp310", "linux_x86_64", "12.8"): (
        "https://github.com/JamePeng/llama-cpp-python/releases/download/"
        "v0.3.35-cu128-Basic-linux-20260406/"
        "llama_cpp_python-0.3.35+cu128.basic-cp310-cp310-linux_x86_64.whl"
    ),
    ("cp311", "linux_x86_64", "12.8"): (
        "https://github.com/JamePeng/llama-cpp-python/releases/download/"
        "v0.3.35-cu128-Basic-linux-20260406/"
        "llama_cpp_python-0.3.35+cu128.basic-cp311-cp311-linux_x86_64.whl"
    ),
    ("cp312", "linux_x86_64", "12.8"): (
        "https://github.com/JamePeng/llama-cpp-python/releases/download/"
        "v0.3.35-cu128-Basic-linux-20260406/"
        "llama_cpp_python-0.3.35+cu128.basic-cp312-cp312-linux_x86_64.whl"
    ),
    ("cp313", "linux_x86_64", "12.8"): (
        "https://github.com/JamePeng/llama-cpp-python/releases/download/"
        "v0.3.35-cu128-Basic-linux-20260406/"
        "llama_cpp_python-0.3.35+cu128.basic-cp313-cp313-linux_x86_64.whl"
    ),
    ("cp314", "linux_x86_64", "12.8"): (
        "https://github.com/JamePeng/llama-cpp-python/releases/download/"
        "v0.3.35-cu128-Basic-linux-20260406/"
        "llama_cpp_python-0.3.35+cu128.basic-cp314-cp314-linux_x86_64.whl"
    ),
    ("cp312", "linux_x86_64", "13.0"): (
        "https://github.com/JamePeng/llama-cpp-python/releases/download/"
        "v0.3.34-cu130-Basic-linux-20260331/"
        "llama_cpp_python-0.3.34+cu130.basic-cp312-cp312-linux_x86_64.whl"
    ),
}
VISION_HANDLER_NAMES = ("Qwen3VLChatHandler", "Qwen25VLChatHandler")
_LAST_BACKEND_INFO: dict | None = None
_PRINTED_BACKEND_LINES: set[str] = set()


class _PathOnlyDllDirectoryHandle:
    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _normalize_bool_env(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _run_command_text(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return ""
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def _normalize_cuda_version(version: str) -> str:
    match = re.search(r"(\d+)\.(\d+)", str(version or ""))
    if not match:
        return ""
    return f"{int(match.group(1))}.{int(match.group(2))}"


def _parse_cuda_version(text: str) -> str:
    if not text:
        return ""
    patterns = (
        r"CUDA Version:\s*(\d+\.\d+)",
        r"release\s+(\d+\.\d+)",
        r"CUDA\s+Version\s+(\d+\.\d+)",
        r"V(\d+\.\d+)(?:\.\d+)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _normalize_cuda_version(match.group(1))
    return ""


def _detect_torch_cuda_version() -> str:
    try:
        import torch

        torch_version = getattr(torch, "version", None)
        return _normalize_cuda_version(str(getattr(torch_version, "cuda", "") or ""))
    except Exception:
        return ""


def _detect_cuda_version() -> str:
    return _detect_torch_cuda_version()


def _detect_cuda_home_version() -> str:
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.getenv(env_name, "").strip()
        if not cuda_home:
            continue
        version_file = os.path.join(cuda_home, "version.txt")
        if os.path.isfile(version_file):
            try:
                with open(version_file, "r", encoding="utf-8", errors="ignore") as handle:
                    parsed = _parse_cuda_version(handle.read())
                if parsed:
                    return parsed
            except Exception:
                pass
        parsed = _normalize_cuda_version(os.path.basename(os.path.normpath(cuda_home)))
        if parsed:
            return parsed

    if platform.system().lower() == "linux":
        for path in ("/usr/local/cuda/version.txt",):
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    parsed = _parse_cuda_version(handle.read())
                if parsed:
                    return parsed
            except Exception:
                pass
    return ""


def _detect_nvcc_cuda_version() -> str:
    return _parse_cuda_version(_run_command_text(["nvcc", "--version"]))


def _detect_nvidia_smi_cuda_version() -> str:
    return _parse_cuda_version(_run_command_text(["nvidia-smi"]))


def _detect_cuda_versions() -> dict[str, str]:
    toolkit = _detect_cuda_home_version() or _detect_nvcc_cuda_version()
    torch_cuda = _detect_torch_cuda_version()
    nvidia_smi = _detect_nvidia_smi_cuda_version()
    selected = toolkit or torch_cuda or nvidia_smi
    selected_source = "toolkit" if toolkit else "torch" if torch_cuda else "nvidia-smi" if nvidia_smi else "unknown"
    return {
        "toolkit": toolkit,
        "torch": torch_cuda,
        "nvidia_smi": nvidia_smi,
        "selected": selected,
        "selected_source": selected_source,
    }


def _resolve_linux_install_spec() -> tuple[str, str]:
    wheel_override = os.getenv("THINKINGLLM_LLAMA_CPP_LINUX_WHEEL_URL", "").strip()
    if wheel_override:
        return wheel_override, "environment wheel override"

    spec_override = os.getenv("THINKINGLLM_LLAMA_CPP_LINUX_SPEC", "").strip()
    if spec_override:
        return spec_override, "environment spec override"

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        platform_tag = "linux_x86_64"
    elif machine in {"aarch64", "arm64"}:
        platform_tag = "linux_aarch64"
    else:
        platform_tag = f"linux_{machine or 'unknown'}"

    cuda_versions = _detect_cuda_versions()
    candidates = []
    for source in ("toolkit", "torch", "nvidia_smi"):
        cuda_key = cuda_versions.get(source) or ""
        if cuda_key and cuda_key not in candidates:
            candidates.append(cuda_key)

    for cuda_key in candidates:
        known_spec = KNOWN_LINUX_WHEEL_SPECS.get((py_tag, platform_tag, cuda_key))
        if known_spec:
            source = next((name for name in ("toolkit", "torch", "nvidia_smi") if cuda_versions.get(name) == cuda_key), "detected")
            return known_spec, f"known Linux wheel ({source} CUDA {cuda_key})"

    if _normalize_bool_env("THINKINGLLM_LLAMA_CPP_ALLOW_SOURCE_BUILD", False):
        return DEFAULT_JAMEPENG_GIT_SPEC, "source-build fallback (explicit opt-in)"

    return "", "no verified Linux wheel match"


def _clear_llama_cpp_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "llama_cpp" or module_name.startswith("llama_cpp."):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _prepend_to_path_once(path: str) -> None:
    current = os.environ.get("PATH", "")
    normalized_path = os.path.normcase(os.path.normpath(path))
    current_entries = [entry for entry in current.split(os.pathsep) if entry]
    for entry in current_entries:
        if os.path.normcase(os.path.normpath(entry)) == normalized_path:
            return
    os.environ["PATH"] = path if not current else path + os.pathsep + current


def _is_windows_path_length_error(exc: OSError) -> bool:
    return platform.system().lower() == "windows" and getattr(exc, "winerror", None) == 206


def _get_windows_runtime_dll_dirs() -> list[str]:
    runtime_dirs: list[str] = []
    try:
        import torch

        torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib_dir):
            runtime_dirs.append(torch_lib_dir)
    except Exception:
        pass

    return runtime_dirs


@contextlib.contextmanager
def relax_windows_dll_directory_for_long_paths():
    if platform.system().lower() != "windows" or not hasattr(os, "add_dll_directory"):
        yield
        return

    original_add_dll_directory = os.add_dll_directory
    path_was_present = "PATH" in os.environ
    original_path = os.environ.get("PATH")

    def _safe_add_dll_directory(path):
        path_str = os.fspath(path)
        try:
            return original_add_dll_directory(path_str)
        except OSError as exc:
            if not _is_windows_path_length_error(exc) or not os.path.isdir(path_str):
                raise
            _prepend_to_path_once(path_str)
            return _PathOnlyDllDirectoryHandle()

    os.add_dll_directory = _safe_add_dll_directory
    try:
        for runtime_dir in _get_windows_runtime_dll_dirs():
            _safe_add_dll_directory(runtime_dir)
            _prepend_to_path_once(runtime_dir)
        yield
    finally:
        if path_was_present:
            os.environ["PATH"] = original_path or ""
        else:
            os.environ.pop("PATH", None)
        os.add_dll_directory = original_add_dll_directory


_relax_windows_dll_directory_for_long_paths = relax_windows_dll_directory_for_long_paths


def _import_llama_cpp_backend(require_vision_handlers: bool):
    global _LAST_BACKEND_INFO
    _clear_llama_cpp_modules()
    with relax_windows_dll_directory_for_long_paths():
        llama_cpp = importlib.import_module("llama_cpp")
        llama_class = getattr(llama_cpp, "Llama", None)
        if llama_class is None:
            raise ImportError("llama_cpp imported but does not expose Llama")

        vision_handlers = []
        if require_vision_handlers:
            chat_format = importlib.import_module("llama_cpp.llama_chat_format")
            vision_handlers = [name for name in VISION_HANDLER_NAMES if hasattr(chat_format, name)]
            if not vision_handlers:
                raise ImportError(
                    "llama_cpp is installed but missing Qwen vision chat handlers "
                    f"({', '.join(VISION_HANDLER_NAMES)})"
                )
        else:
            try:
                chat_format = importlib.import_module("llama_cpp.llama_chat_format")
                vision_handlers = [name for name in VISION_HANDLER_NAMES if hasattr(chat_format, name)]
            except Exception:
                vision_handlers = []

        _LAST_BACKEND_INFO = _collect_llama_cpp_backend_info(llama_cpp, vision_handlers)
        _print_backend_info(_LAST_BACKEND_INFO)

    return llama_class


def _probe_gpu_offload_support() -> bool | None:
    try:
        low_level = importlib.import_module("llama_cpp.llama_cpp")
    except Exception:
        return None

    candidates = [low_level]
    lib = getattr(low_level, "_lib", None)
    if lib is not None:
        candidates.append(lib)

    for candidate in candidates:
        probe = getattr(candidate, "llama_supports_gpu_offload", None)
        if probe is None:
            continue
        try:
            return bool(probe())
        except Exception:
            return None
    return None


def _package_version() -> str:
    try:
        return importlib.metadata.version("llama-cpp-python")
    except Exception:
        return "unknown"


def _collect_llama_cpp_backend_info(llama_cpp, vision_handlers: list[str]) -> dict:
    cuda_versions = _detect_cuda_versions()
    return {
        "version": str(getattr(llama_cpp, "__version__", "") or _package_version()),
        "module_path": str(getattr(llama_cpp, "__file__", "") or "unknown"),
        "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "platform": platform.platform(),
        "machine": platform.machine().lower(),
        "torch_cuda": cuda_versions.get("torch", ""),
        "toolkit_cuda": cuda_versions.get("toolkit", ""),
        "nvidia_smi_cuda": cuda_versions.get("nvidia_smi", ""),
        "selected_cuda": cuda_versions.get("selected", ""),
        "selected_cuda_source": cuda_versions.get("selected_source", "unknown"),
        "vision_handlers": list(vision_handlers),
        "gpu_offload": _probe_gpu_offload_support(),
    }


def _format_bool_status(value) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def format_llama_cpp_backend_info(info: dict | None = None) -> str:
    info = info or _LAST_BACKEND_INFO or {}
    handlers = ",".join(info.get("vision_handlers") or []) or "missing"
    return (
        f"llama-cpp-python {info.get('version', 'unknown')} "
        f"({info.get('python_tag', 'unknown')}, {info.get('machine', 'unknown')}); "
        f"GPU offload={_format_bool_status(info.get('gpu_offload'))}; "
        f"vision={handlers}; "
        f"CUDA torch={info.get('torch_cuda') or 'unknown'}, "
        f"toolkit={info.get('toolkit_cuda') or 'unknown'}, "
        f"nvidia-smi={info.get('nvidia_smi_cuda') or 'unknown'}; "
        f"module={info.get('module_path', 'unknown')}"
    )


def _print_backend_info(info: dict) -> None:
    line = f"[QwenVL] llama.cpp backend: {format_llama_cpp_backend_info(info)}"
    if line in _PRINTED_BACKEND_LINES:
        return
    _PRINTED_BACKEND_LINES.add(line)
    print(line)


def get_last_llama_cpp_backend_info() -> dict | None:
    return dict(_LAST_BACKEND_INFO) if _LAST_BACKEND_INFO else None


def _build_source_install_env() -> dict[str, str]:
    install_env = dict(os.environ)
    toolkit_cuda = _detect_cuda_home_version() or _detect_nvcc_cuda_version()
    if toolkit_cuda:
        existing_args = install_env.get("CMAKE_ARGS", "").strip()
        cuda_arg = "-DGGML_CUDA=on"
        install_env["CMAKE_ARGS"] = f"{existing_args} {cuda_arg}".strip() if cuda_arg not in existing_args else existing_args
        install_env["FORCE_CMAKE"] = "1"
        return install_env
    if not _normalize_bool_env("THINKINGLLM_LLAMA_CPP_ALLOW_CPU_SOURCE_BUILD", False):
        raise RuntimeError(
            "[QwenVL] Refusing automatic llama_cpp source build because no CUDA toolkit was detected. "
            "Set THINKINGLLM_LLAMA_CPP_LINUX_WHEEL_URL to a matching wheel, install CUDA toolkit, or set "
            "THINKINGLLM_LLAMA_CPP_ALLOW_CPU_SOURCE_BUILD=1 if you intentionally want a CPU build."
        )
    return install_env


def ensure_llama_cpp_backend(require_vision_handlers: bool = False):
    try:
        return _import_llama_cpp_backend(require_vision_handlers=require_vision_handlers)
    except Exception as first_exc:
        if platform.system().lower() != "linux" or not _normalize_bool_env("THINKINGLLM_AUTO_INSTALL_LLAMA_CPP", True):
            raise RuntimeError(
                "[QwenVL] llama_cpp is unavailable or incompatible. Install the GGUF vision dependency first. "
                "See docs/LLAMA_CPP_PYTHON_VISION_INSTALL.md"
            ) from first_exc

        install_spec, install_reason = _resolve_linux_install_spec()
        if not install_spec:
            cuda_versions = _detect_cuda_versions()
            raise RuntimeError(
                "[QwenVL] llama_cpp is unavailable or incompatible, and no verified JamePeng Linux wheel matches this runtime. "
                f"Detected CUDA: toolkit={cuda_versions.get('toolkit') or 'unknown'}, "
                f"torch={cuda_versions.get('torch') or 'unknown'}, nvidia-smi={cuda_versions.get('nvidia_smi') or 'unknown'}. "
                "Set THINKINGLLM_LLAMA_CPP_LINUX_WHEEL_URL to a matching wheel, "
                "THINKINGLLM_LLAMA_CPP_LINUX_SPEC to an explicit package spec, or "
                "THINKINGLLM_LLAMA_CPP_ALLOW_SOURCE_BUILD=1 to opt into a local source build."
            ) from first_exc
        install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            install_spec,
        ]
        print(
            f"[QwenVL] llama_cpp check failed on Linux; attempting automatic install "
            f"using {install_reason}: {install_spec}"
        )
        install_env = _build_source_install_env() if install_spec == DEFAULT_JAMEPENG_GIT_SPEC else None
        result = subprocess.run(install_cmd, capture_output=True, text=True, env=install_env)
        if result.returncode != 0:
            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                "[QwenVL] Automatic llama_cpp installation failed on Linux. "
                f"Command: {' '.join(install_cmd)}\nstdout:\n{stdout or '<empty>'}\nstderr:\n{stderr or '<empty>'}"
            ) from first_exc

        try:
            return _import_llama_cpp_backend(require_vision_handlers=require_vision_handlers)
        except Exception as second_exc:
            raise RuntimeError(
                "[QwenVL] llama_cpp was installed, but the backend is still incompatible. "
                "Set THINKINGLLM_LLAMA_CPP_LINUX_WHEEL_URL to a matching JamePeng wheel or "
                "THINKINGLLM_LLAMA_CPP_LINUX_SPEC to an explicit package spec, then restart ComfyUI."
            ) from second_exc