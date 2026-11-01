"""Constrain workflow-provided audio inputs and their processing cost."""

import math
import os
from pathlib import Path


MAX_AUDIO_SECONDS = 600
MAX_AUDIO_FILE_BYTES = 256 * 1024 * 1024
FFMPEG_TIMEOUT_SECONDS = 120
MAX_AUDIO_SAMPLE_RATE = 192_000
MAX_AUDIO_ELEMENTS = MAX_AUDIO_SECONDS * 48_000 * 2
AUDIO_ALLOWED_ROOTS_ENV = "THINKINGLLM_AUDIO_ALLOWED_ROOTS"


def _configured_roots(default_roots=None) -> list[Path]:
    if default_roots is None:
        try:
            import folder_paths

            default_roots = [folder_paths.get_input_directory()]
        except Exception as exc:
            raise RuntimeError("Unable to determine ComfyUI's input directory for audio path validation") from exc

    roots: list[Path] = []
    configured = list(default_roots or [])
    extra_roots = os.environ.get(AUDIO_ALLOWED_ROOTS_ENV, "")
    configured.extend(root for root in extra_roots.split(os.pathsep) if root.strip())

    for value in configured:
        root = Path(os.path.expandvars(str(value))).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Configured audio root is not a directory: {root}")
        if root not in roots:
            roots.append(root)

    if not roots:
        raise RuntimeError("No audio input directories are configured")
    return roots


def resolve_local_audio_path(audio_file_path, *, default_roots=None) -> Path | None:
    """Resolve an audio path only when its canonical target is in an allowed root."""
    if audio_file_path is None:
        return None
    raw_path = str(audio_file_path).strip().strip("\"'")
    if not raw_path:
        return None

    roots = _configured_roots(default_roots)
    candidate = Path(os.path.expandvars(raw_path)).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [root / candidate for root in roots] + [candidate]

    for path in candidates:
        resolved = path.resolve(strict=False)
        if any(resolved == root or root in resolved.parents for root in roots):
            return resolved

    allowed = ", ".join(str(root) for root in roots)
    raise ValueError(f"audio_file_path must resolve inside an allowed audio directory: {allowed}")


def validate_audio_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"audio_file_path does not exist or is not a file: {path}")
    if path.stat().st_size > MAX_AUDIO_FILE_BYTES:
        raise ValueError(f"audio_file_path exceeds the {MAX_AUDIO_FILE_BYTES // (1024 * 1024)} MiB size limit")


def validate_audio_duration(sample_count: int, sample_rate: int) -> None:
    if sample_rate <= 0:
        raise ValueError(f"invalid sample_rate={sample_rate!r}")
    if sample_rate > MAX_AUDIO_SAMPLE_RATE:
        raise ValueError(f"audio sample rate exceeds the {MAX_AUDIO_SAMPLE_RATE} Hz limit")
    if sample_count < 0 or sample_count > MAX_AUDIO_SECONDS * sample_rate:
        raise ValueError(f"audio input exceeds the {MAX_AUDIO_SECONDS}-second duration limit")


def validate_audio_shape(shape, sample_rate: int) -> None:
    dimensions = tuple(int(value) for value in shape)
    if not dimensions or any(value < 0 for value in dimensions):
        raise ValueError("invalid audio waveform shape")
    validate_audio_duration(max(dimensions), sample_rate)
    if math.prod(dimensions) > MAX_AUDIO_ELEMENTS:
        raise ValueError("audio waveform exceeds the decoded sample limit")
