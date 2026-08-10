"""Prompt budgeting plus duration-aware video prompt contracts."""

import math
import re

QWEN_MIN_REASONING_BUDGET_TOKENS = 1024
QWEN_FINAL_ANSWER_RESERVE_TOKENS = 256
QWEN_CONTEXT_OVERHEAD_TOKENS = 128
_QWEN_SOFT_SWITCHES = {"/think", "/no_think", "/nothink"}
MINIMAX_H3_REFERENCE_PRESET = "🖼️ MiniMax H3 Reference-to-Video"
VIDEO_PROMPT_CONTRACT_VERSION = 10
_MINIMAX_BASE_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_MINIMAX_REFERENCE_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
VIDEO_DURATION_INPUT = (
    "FLOAT",
    {
        "default": 5.0,
        "min": 0.2,
        "max": 150.0,
        "step": 0.1,
        "tooltip": (
            "Target video duration in seconds. It is used only by registered LTX 2.3 and MiniMax H3 "
            "video presets. Connect the same requested duration to the video generator; MiniMax values "
            "are normalized to its 17k+5 frame grid at 24 fps. For longer MiniMax scripts, ThinkingLLM "
            "selects a coherent moment that fits while keeping any selected dialogue verbatim."
        ),
    },
)
VIDEO_PRESET_METADATA: dict[str, dict] = {}


def estimate_qwen_text_tokens(*parts):
    """Conservatively estimate prompt tokens when an exact tokenizer is unavailable."""
    total_chars = 0
    non_empty_parts = 0
    for part in parts:
        if not part:
            continue
        normalized = " ".join(str(part).split())
        if not normalized:
            continue
        total_chars += len(normalized)
        non_empty_parts += 1
    if total_chars <= 0:
        return 0
    return max(1, (total_chars + 3) // 4 + max(0, non_empty_parts - 1))


def resolve_qwen_context_window(config, fallback=32768):
    """Return the best available HF context window from the model config."""
    candidates = []
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is None:
            continue
        for attr in ("max_position_embeddings", "n_positions", "seq_length"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and 0 < value <= 2_000_000:
                candidates.append(value)
    if candidates:
        return max(candidates)
    return int(fallback)


def resolve_qwen_thinking_mode(
    requested_enable,
    max_tokens,
    label="QwenVL",
    *,
    prompt_tokens=None,
    context_window=None,
    min_reasoning_tokens=QWEN_MIN_REASONING_BUDGET_TOKENS,
    answer_reserve_tokens=QWEN_FINAL_ANSWER_RESERVE_TOKENS,
    context_overhead_tokens=QWEN_CONTEXT_OVERHEAD_TOKENS,
    quiet=False,
):
    """Enable thinking only when requested and enough generation budget remains."""
    if not bool(requested_enable):
        return False
    requested_output = max(0, int(max_tokens or 0))
    available_output = requested_output
    prompt_token_count = max(0, int(prompt_tokens or 0))
    context_limit = int(context_window) if context_window is not None else None

    if context_limit is not None:
        remaining_context = max(0, context_limit - prompt_token_count - int(context_overhead_tokens))
        available_output = min(available_output, remaining_context) if available_output else remaining_context

    minimum_required = int(min_reasoning_tokens) + int(answer_reserve_tokens)
    if available_output <= 0 or available_output < minimum_required:
        if context_limit is not None:
            budget_note = (
                f"max_tokens={requested_output}, prompt_tokens={prompt_token_count}, "
                f"context_window={context_limit}, available_output={available_output}"
            )
        else:
            budget_note = f"max_tokens={requested_output}, available_output={available_output}"
        if not quiet:
            print(
                f"[{label}] Thinking requested but {budget_note} is below the required {minimum_required} tokens "
                f"({int(min_reasoning_tokens)} reasoning + {int(answer_reserve_tokens)} answer reserve); "
                "forcing non-thinking mode to preserve answer space."
            )
        return False
    return True


def apply_qwen_soft_thinking_directive(prompt_text, enable_thinking, supports_soft_switch=False):
    """Append the latest Qwen3 soft switch when the backend supports it."""
    text = prompt_text or ""
    if not supports_soft_switch:
        return text
    lines = [line for line in text.splitlines() if line.strip().lower() not in _QWEN_SOFT_SWITCHES]
    lines.append("/think" if enable_thinking else "/no_think")
    return "\n".join(lines).strip()


def validate_minimax_reference_request(preset_prompt, custom_prompt):
    """Require an explicit target and reference role for MiniMax H3 R2V prompting."""
    if preset_prompt == MINIMAX_H3_REFERENCE_PRESET and not (custom_prompt or "").strip():
        raise ValueError(
            "[QwenVL] MiniMax H3 Reference-to-Video requires a custom prompt describing the target video "
            "and how each reference should be used. To animate a single image as the first frame, use "
            "MiniMax H3 Text-to-Video instead."
        )


def get_video_preset_metadata(preset_name):
    """Return duration metadata only for a registered duration-aware video preset."""
    metadata = VIDEO_PRESET_METADATA.get(preset_name)
    if not isinstance(metadata, dict) or not metadata.get("duration_required"):
        return None
    return metadata


def resolve_video_duration(preset_name, duration_seconds=5.0):
    """Resolve an optional node duration to the provider-native effective duration."""
    metadata = get_video_preset_metadata(preset_name)
    if metadata is None:
        return None
    try:
        requested_seconds = float(5.0 if duration_seconds is None else duration_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("[QwenVL] duration_seconds must be a finite number between 0.2 and 150.0.") from exc
    if not math.isfinite(requested_seconds) or not 0.2 <= requested_seconds <= 150.0:
        raise ValueError("[QwenVL] duration_seconds must be a finite number between 0.2 and 150.0.")

    provider = metadata.get("provider")
    frames = None
    effective_seconds = requested_seconds
    if provider == "minimax_h3":
        frames = max(5, round(requested_seconds * 24))
        frames += (5 - (frames % 17)) % 17
        effective_seconds = frames / 24
        duration_text = f"{effective_seconds:.2f}"
    elif provider == "ltx_2_3":
        duration_text = f"{effective_seconds:.2f}".rstrip("0").rstrip(".")
    else:
        return None

    return {
        "provider": provider,
        "video_mode": metadata.get("video_mode"),
        "requested_seconds": requested_seconds,
        "effective_seconds": effective_seconds,
        "duration_text": duration_text,
        "frames": frames,
        "contract_version": VIDEO_PROMPT_CONTRACT_VERSION if provider == "minimax_h3" else None,
    }


def apply_video_duration_context(prompt_text, preset_name, duration_seconds=5.0):
    """Append authoritative duration context for the LLM without changing non-video presets."""
    resolved = resolve_video_duration(preset_name, duration_seconds)
    if resolved is None:
        return prompt_text
    duration_text = resolved["duration_text"]
    provider_instruction = (
        f'Express the total duration naturally as a "{duration_text}-second continuous shot" in the single paragraph.'
        if resolved["provider"] == "ltx_2_3"
        else (
            "Use this duration only to plan feasible actions, bound every cut timestamp, and fill any "
            "official FL2VA/L2VA alignment line. Do not print TARGET_DURATION_SECONDS or a separate "
            "target-duration declaration in the final MiniMax prompt."
        )
    )
    duration_context = (
        f"TARGET_DURATION_SECONDS: {duration_text}\n"
        "This node value is authoritative and overrides any conflicting duration in the user text. "
        + ("Mention the total target duration exactly once in the final prompt. " if resolved["provider"] == "ltx_2_3" else "")
        + provider_instruction
    )
    return "\n\n".join(part for part in ((prompt_text or "").strip(), duration_context) if part)


def _normalize_minimax_dialogue_text(text):
    return " ".join((text or "").split())


_MINIMAX_LANGUAGE_LABELS = (
    "English", "German", "French", "Spanish", "Italian", "Portuguese", "Dutch", "Polish",
    "Russian", "Ukrainian", "Czech", "Slovak", "Swedish", "Norwegian", "Danish", "Finnish",
    "Greek", "Turkish", "Arabic", "Hebrew", "Hindi", "Bengali", "Urdu", "Persian", "Thai",
    "Vietnamese", "Indonesian", "Malay", "Chinese", "Mandarin", "Cantonese", "Japanese", "Korean",
)


def _normalize_minimax_dialogue_tags(text):
    """Normalize safe language-tag variants to official <d>[Language] ...</d> syntax."""
    language_pattern = "|".join(re.escape(language) for language in _MINIMAX_LANGUAGE_LABELS)

    def _normalize_tag(match):
        language = match.group(1).strip()
        payload = match.group(2).strip()
        return f"<d>[{language}] {payload}</d>"

    normalized = re.sub(
        r"(?is)<d>\s*\[\s*([^\]\r\n]+?)\s*\]\s*(.+?)\s*</d>",
        _normalize_tag,
        text,
    )
    return re.sub(
        rf"(?is)<d>\s*({language_pattern})\s*:?[ \t\r\n]+(.+?)\s*</d>",
        _normalize_tag,
        normalized,
    )


def extract_minimax_source_dialogue(source_prompt_text):
    """Extract explicitly quoted, standalone dialogue blocks from a text request."""
    source = (source_prompt_text or "").strip()
    if not source:
        return []

    dialogue = []
    tagged = re.findall(r"<d>\[[^\]\r\n]+\]\s*(.+?)</d>", source, flags=re.DOTALL | re.IGNORECASE)
    dialogue.extend(_normalize_minimax_dialogue_text(item) for item in tagged if item.strip())

    quote_pairs = (("“", "”"), ("„", "“"), ("«", "»"), ('"', '"'))
    for paragraph in re.split(r"\r?\n[ \t]*\r?\n", source):
        candidate = paragraph.strip()
        if not candidate or candidate.lower().startswith("<d>"):
            continue
        for opening, closing in quote_pairs:
            if len(candidate) >= 2 and candidate.startswith(opening) and candidate.endswith(closing):
                payload = _normalize_minimax_dialogue_text(candidate[len(opening):-len(closing)])
                if payload:
                    dialogue.append(payload)
                break

    unique = []
    for item in dialogue:
        if item not in unique:
            unique.append(item)
    return unique


def validate_minimax_source_duration(preset_name, duration_seconds=5.0, source_prompt_text=None):
    """Return source dialogue for MiniMax presets without rejecting longer master scripts."""
    resolved = resolve_video_duration(preset_name, duration_seconds)
    if resolved is None or resolved["provider"] != "minimax_h3":
        return []
    return extract_minimax_source_dialogue(source_prompt_text)


def _normalize_minimax_section_boundaries(text, expected_fields):
    """Put uniquely identifiable official fields on their own lines without rewriting content."""
    found = []
    for field in expected_fields:
        matches = list(re.finditer(rf"(?i)(?<![A-Za-z0-9_]){re.escape(field)}[ \t]*:", text))
        if len(matches) != 1:
            return text
        found.append((field, matches[0]))
    if [match.start() for _, match in found] != sorted(match.start() for _, match in found):
        return text

    normalized = text
    for _, match in reversed(found[1:]):
        start = match.start()
        normalized = normalized[:start].rstrip() + "\n\n" + normalized[start:].lstrip()
    return normalized


def _normalize_minimax_shot_syntax(description):
    """Normalize only unambiguous MiniMax shot syntax without validating the timeline."""
    timestamp_pattern = re.compile(
        r"(\[Shot\s+(\d+)\]\s+At\s+)(\d{1,2}):(\d{1,2})\.(\d{1,3})\s*,?",
        flags=re.IGNORECASE,
    )

    def _normalize_timestamp(match):
        minutes = int(match.group(3))
        seconds = int(match.group(4))
        millis = int(match.group(5).ljust(3, "0"))
        if seconds >= 60:
            return match.group(0)
        return f"[Shot {int(match.group(2))}] At {minutes:02d}:{seconds:02d}.{millis:03d},"

    normalized = timestamp_pattern.sub(_normalize_timestamp, description)
    normalized = re.sub(
        r"(?i)\[Shot\s+1\]\s+At\s+00:00\.000,?[ \t]*",
        "[Shot 1] ",
        normalized,
        count=1,
    )
    return normalized


def normalize_minimax_prompt_best_effort(prompt_text, preset_name, duration_seconds=5.0):
    """Apply safe MiniMax formatting fixes without rejecting or repairing model output."""
    resolved = resolve_video_duration(preset_name, duration_seconds)
    if resolved is None or resolved["provider"] != "minimax_h3":
        return prompt_text

    original = (prompt_text or "").strip()
    if not original:
        return original

    try:
        text = re.sub(
            r"(?i)\bTarget\s+duration\s*:\s*\d+(?:[.,]\d+)?\s*seconds?\.?[ \t]*",
            "",
            original,
        ).strip()
        text = _normalize_minimax_dialogue_tags(text)
        expected_fields = (
            _MINIMAX_REFERENCE_FIELDS
            if resolved["video_mode"] == "ref2va"
            else _MINIMAX_BASE_FIELDS
        )
        text = _normalize_minimax_section_boundaries(text, expected_fields)
        text = _normalize_minimax_shot_syntax(text)
        return text.strip() or original
    except Exception:
        # Formatting assistance must never become a second schema gate in front of H3.
        return original


def ensure_video_prompt_duration(prompt_text, preset_name, duration_seconds=5.0, source_prompt_text=None):
    """Apply provider-native duration guidance and safe output normalization."""
    resolved = resolve_video_duration(preset_name, duration_seconds)
    if resolved is None:
        return prompt_text
    text = (prompt_text or "").strip()
    duration_text = resolved["duration_text"]

    if resolved["provider"] == "ltx_2_3":
        text = " ".join(text.split())
        marker = f"{duration_text}-second continuous shot"
        number_pattern = re.escape(duration_text).replace(r"\.", r"[.,]")
        marker_pattern = re.compile(
            rf"\b{number_pattern}(?:\.0+)?[- ]second continuous shot\b",
            flags=re.IGNORECASE,
        )
        seen = False

        def _canonicalize_ltx_marker(_match):
            nonlocal seen
            if seen:
                return "continuous shot"
            seen = True
            return marker

        text = marker_pattern.sub(_canonicalize_ltx_marker, text)
        if not seen:
            text = f"A {marker}. {text}".strip()
        return " ".join(text.split())

    if resolved["provider"] == "minimax_h3":
        return normalize_minimax_prompt_best_effort(
            text,
            preset_name,
            duration_seconds,
        )

    return text
