import importlib
import os
import platform
import subprocess
import sys

DEFAULT_JAMEPENG_GIT_SPEC = "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git"
KNOWN_LINUX_WHEEL_SPECS = {
    ("cp312", "linux_x86_64", "13.0"): (
        "https://github.com/JamePeng/llama-cpp-python/releases/download/"
        "v0.3.34-cu130-Basic-linux-20260331/"
        "llama_cpp_python-0.3.34+cu130.basic-cp312-cp312-linux_x86_64.whl"
    ),
}
VISION_HANDLER_NAMES = ("Qwen3VLChatHandler", "Qwen25VLChatHandler")


def _normalize_bool_env(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _detect_cuda_version() -> str:
    try:
        import torch

        return str(getattr(torch.version, "cuda", "") or "")
    except Exception:
        return ""


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

    cuda_version = _detect_cuda_version()
    cuda_key = ".".join(cuda_version.split(".")[:2]) if cuda_version else ""
    known_spec = KNOWN_LINUX_WHEEL_SPECS.get((py_tag, platform_tag, cuda_key))
    if known_spec:
        return known_spec, "known Linux wheel"

    return DEFAULT_JAMEPENG_GIT_SPEC, "source-build fallback"


def _clear_llama_cpp_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "llama_cpp" or module_name.startswith("llama_cpp."):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _import_llama_cpp_backend(require_vision_handlers: bool):
    _clear_llama_cpp_modules()
    llama_cpp = importlib.import_module("llama_cpp")
    llama_class = getattr(llama_cpp, "Llama", None)
    if llama_class is None:
        raise ImportError("llama_cpp imported but does not expose Llama")

    if require_vision_handlers:
        chat_format = importlib.import_module("llama_cpp.llama_chat_format")
        if not any(hasattr(chat_format, name) for name in VISION_HANDLER_NAMES):
            raise ImportError(
                "llama_cpp is installed but missing Qwen vision chat handlers "
                f"({', '.join(VISION_HANDLER_NAMES)})"
            )

    return llama_class


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
        result = subprocess.run(install_cmd, capture_output=True, text=True)
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