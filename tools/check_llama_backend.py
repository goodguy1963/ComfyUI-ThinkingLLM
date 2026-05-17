"""Check the ThinkingLLM llama-cpp-python backend outside ComfyUI.

Examples:
    python tools/check_llama_backend.py
    python tools/check_llama_backend.py --model /path/to/model.gguf --gpu-layers 1 --verbose-load
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from AILab_LlamaCppInstaller import (  # noqa: E402
    ensure_llama_cpp_backend,
    format_llama_cpp_backend_info,
    get_last_llama_cpp_backend_info,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ThinkingLLM llama.cpp backend health.")
    parser.add_argument("--model", help="Optional GGUF model path for a load smoke test.")
    parser.add_argument("--gpu-layers", type=int, default=1, help="GPU layers for the optional model smoke test.")
    parser.add_argument("--ctx", type=int, default=512, help="Context size for the optional model smoke test.")
    parser.add_argument("--batch", type=int, default=128, help="Batch size for the optional model smoke test.")
    parser.add_argument("--no-vision", action="store_true", help="Do not require Qwen vision chat handlers.")
    parser.add_argument("--verbose-load", action="store_true", help="Print verbose llama.cpp load logs for the smoke test.")
    parser.add_argument("--strict-gpu", action="store_true", help="Exit nonzero if GPU offload cannot be confirmed.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        Llama = ensure_llama_cpp_backend(require_vision_handlers=not args.no_vision)
    except Exception as exc:
        print(f"BACKEND_IMPORT=failed\n{exc}", file=sys.stderr)
        return 2

    info = get_last_llama_cpp_backend_info() or {}
    print("BACKEND_IMPORT=ok")
    print(format_llama_cpp_backend_info(info))

    gpu_offload = info.get("gpu_offload")
    if args.strict_gpu and gpu_offload is not True:
        print("STRICT_GPU=failed: backend did not confirm GPU offload support", file=sys.stderr)
        return 3

    if args.model:
        model_path = Path(args.model).expanduser().resolve()
        if not model_path.exists():
            print(f"MODEL_LOAD=failed: file not found: {model_path}", file=sys.stderr)
            return 4
        previous_verbose = os.environ.get("THINKINGLLM_LLAMA_CPP_VERBOSE_LOAD")
        if args.verbose_load:
            os.environ["THINKINGLLM_LLAMA_CPP_VERBOSE_LOAD"] = "1"
        try:
            print(
                f"MODEL_LOAD=starting model={model_path.name} gpu_layers={args.gpu_layers} "
                f"ctx={args.ctx} batch={args.batch}"
            )
            llm = Llama(
                model_path=str(model_path),
                n_gpu_layers=int(args.gpu_layers),
                n_ctx=int(args.ctx),
                n_batch=int(args.batch),
                verbose=bool(args.verbose_load),
            )
            try:
                output = llm("Hello", max_tokens=4, temperature=0.0)
                print("MODEL_INFERENCE=ok")
                print(str(output)[:500])
            finally:
                close = getattr(llm, "close", None)
                if callable(close):
                    close()
        except Exception as exc:
            print(f"MODEL_LOAD=failed\n{exc}", file=sys.stderr)
            return 5
        finally:
            if previous_verbose is None:
                os.environ.pop("THINKINGLLM_LLAMA_CPP_VERBOSE_LOAD", None)
            else:
                os.environ["THINKINGLLM_LLAMA_CPP_VERBOSE_LOAD"] = previous_verbose

    return 0


if __name__ == "__main__":
    raise SystemExit(main())