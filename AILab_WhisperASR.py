import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


WHISPER_MODEL_CHOICES = [
    "small",
    "base",
    "tiny",
    "medium",
    "large-v3",
    "distil-large-v3",
]

WHISPER_LANGUAGE_CHOICES = [
    "auto",
    "en",
    "de",
    "fr",
    "es",
    "it",
    "pt",
    "nl",
    "pl",
    "tr",
    "ja",
    "ko",
    "zh",
]

WHISPER_INSTALL_MESSAGE = (
    "[ThinkingLLM Whisper ASR] faster-whisper is not installed in this ComfyUI Python. "
    "Install it in the active ComfyUI Python with: python -m pip install faster-whisper"
)


def _clean_audio_file_path(audio_file_path) -> Path | None:
    if audio_file_path is None:
        return None
    raw_path = str(audio_file_path).strip().strip("\"'")
    if not raw_path:
        return None
    return Path(os.path.expandvars(raw_path)).expanduser()


def _ffmpeg_path() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH; install FFmpeg or connect a decoded ComfyUI AUDIO input")
    return ffmpeg


def _decode_audio_file_to_wav(audio_file_path, target_path: Path) -> str:
    source_path = _clean_audio_file_path(audio_file_path)
    if source_path is None:
        raise ValueError("audio_file_path is empty")
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"audio_file_path does not exist or is not a file: {source_path}")

    result = subprocess.run(
        [
            _ffmpeg_path(),
            "-hide_banner",
            "-v",
            "error",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(target_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode audio_file_path: {result.stderr.strip() or 'unknown ffmpeg error'}")
    return str(source_path)


def _audio_to_wav_file(audio, target_path: Path) -> str:
    if not isinstance(audio, dict):
        raise ValueError("expected ComfyUI AUDIO dict with waveform/sample_rate")
    waveform = audio.get("waveform")
    if waveform is None:
        raise ValueError("missing waveform")
    sample_rate = int(audio.get("sample_rate") or 0)
    if sample_rate <= 0:
        raise ValueError(f"invalid sample_rate={audio.get('sample_rate')!r}")

    tensor = waveform.detach().cpu() if hasattr(waveform, "detach") else waveform
    tensor = tensor.cpu() if hasattr(tensor, "cpu") else tensor
    array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        raise ValueError("empty waveform")
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"unsupported waveform shape={getattr(array, 'shape', None)}")
    if array.shape[0] > array.shape[-1]:
        array = array.T
    if not np.all(np.isfinite(array)):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)

    mono = np.mean(array, axis=0).astype(np.float32)
    if sample_rate != 16000:
        target_length = max(1, int(round(float(mono.size) * 16000.0 / float(sample_rate))))
        old_positions = np.linspace(0.0, float(mono.size - 1), num=mono.size, dtype=np.float64)
        new_positions = np.linspace(0.0, float(mono.size - 1), num=target_length, dtype=np.float64)
        mono = np.interp(new_positions, old_positions, mono).astype(np.float32)

    pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(target_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm16.tobytes())
    return "ComfyUI AUDIO input"


def _resolve_whisper_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def _resolve_compute_type(device: str, compute_type: str) -> str:
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _missing_audio_result(notes: list[str] | None = None) -> tuple[str, str, str]:
    message = (
        "[ThinkingLLM Whisper ASR] No usable audio input. Connect a ComfyUI AUDIO output "
        "or set audio_file_path to an existing M4A/MP3/WAV/FLAC file."
    )
    trace_lines = ["[AUDIO]", message]
    if notes:
        trace_lines.extend(str(note) for note in notes if note)
    return message, "[]", "\n".join(trace_lines)


class ThinkingLLM_Whisper_ASR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_size": (WHISPER_MODEL_CHOICES, {"default": "small", "tooltip": "faster-whisper model size. small is a reliable Windows-friendly starting point; large-v3 is higher quality but downloads and runs much larger."}),
                "language": (WHISPER_LANGUAGE_CHOICES, {"default": "auto", "tooltip": "Audio language. Use auto to let Whisper detect it."}),
                "task": (["transcribe", "translate"], {"default": "transcribe", "tooltip": "transcribe preserves the original language; translate returns English."}),
                "device": (["cpu", "auto", "cuda"], {"default": "cpu", "tooltip": "cpu is the most reliable Windows default; choose cuda when your faster-whisper/CTranslate2 CUDA runtime is working."}),
                "compute_type": (["int8", "auto", "float16", "int8_float16", "float32"], {"default": "int8", "tooltip": "int8 is reliable on CPU. Use float16 or int8_float16 for CUDA."}),
                "beam_size": ("INT", {"default": 5, "min": 1, "max": 10, "tooltip": "Higher values can improve accuracy but run slower."}),
                "vad_filter": ("BOOLEAN", {"default": True, "tooltip": "Skip long silence and non-speech sections before transcription."}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_file_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional local audio file path. M4A, MP3, WAV, FLAC, and other FFmpeg-readable files are decoded to 16 kHz mono WAV before Whisper."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("TRANSCRIPT", "SEGMENTS_JSON", "RAW_TRACE")
    FUNCTION = "process"
    CATEGORY = "ThinkingLLM/Audio"

    def process(
        self,
        model_size: str,
        language: str,
        task: str,
        device: str,
        compute_type: str,
        beam_size: int,
        vad_filter: bool,
        audio=None,
        audio_file_path="",
    ):
        notes: list[str] = []
        clean_path = _clean_audio_file_path(audio_file_path)
        if audio is None and clean_path is None:
            return _missing_audio_result()

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            trace = "\n".join(["[WHISPER]", WHISPER_INSTALL_MESSAGE, f"import_error={exc}"])
            return WHISPER_INSTALL_MESSAGE, "[]", trace

        with tempfile.TemporaryDirectory(prefix="thinkingllm-whisper-") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            try:
                if clean_path is not None:
                    source_label = _decode_audio_file_to_wav(str(clean_path), wav_path)
                else:
                    source_label = _audio_to_wav_file(audio, wav_path)
            except Exception as exc:
                return _missing_audio_result([str(exc)])

            resolved_device = _resolve_whisper_device(str(device))
            resolved_compute_type = _resolve_compute_type(resolved_device, str(compute_type))
            model = WhisperModel(str(model_size), device=resolved_device, compute_type=resolved_compute_type)
            segments_iter, info = model.transcribe(
                str(wav_path),
                language=None if language == "auto" else str(language),
                task=str(task),
                beam_size=int(beam_size),
                vad_filter=bool(vad_filter),
            )
            segments = [
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": str(segment.text).strip(),
                }
                for segment in segments_iter
            ]

        transcript = "\n".join(segment["text"] for segment in segments if segment["text"]).strip()
        segments_json = json.dumps(segments, ensure_ascii=False, indent=2)
        detected_language = getattr(info, "language", None)
        detected_probability = getattr(info, "language_probability", None)
        duration = getattr(info, "duration", None)
        trace = "\n".join(
            [
                "[WHISPER]",
                f"source={source_label}",
                f"model_size={model_size}; device={resolved_device}; compute_type={resolved_compute_type}; task={task}; vad_filter={bool(vad_filter)}",
                f"language={detected_language}; language_probability={detected_probability}; duration={duration}",
                f"segments={len(segments)}; transcript_chars={len(transcript)}",
            ]
        )
        return transcript, segments_json, trace


NODE_CLASS_MAPPINGS = {
    "ThinkingLLM_Whisper_ASR": ThinkingLLM_Whisper_ASR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ThinkingLLM_Whisper_ASR": "ThinkingLLM Whisper ASR",
}
