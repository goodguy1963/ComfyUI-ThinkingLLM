import json
import re
from dataclasses import dataclass

OUTPUT_CLEANER_VERSION = 5


@dataclass(frozen=True)
class OutputCleanConfig:
    mode: str = "prompt"
    strip_think: bool = True
    strip_code_fences: bool = True
    strip_role_prefixes: bool = True
    strip_json_wrappers: bool = True
    strip_leading_preamble: bool = True
    strip_planning: bool = True
    keep_first_paragraph_only: bool = False


_ROLE_PREFIX_RE = re.compile(r"^\s*(assistant|final|output|response|result|prompt)\s*:\s*", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^\s*```[\w-]*\s*$", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think[^>]*>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think[^>]*>", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)
_CHANNEL_OPEN_RE = re.compile(
    r"(?:<\|?start\|?>\s*assistant\s*)?"
    r"<\|?channel\|?>\s*(?P<channel>analysis|thinking|reasoning|final)\s*"
    r"<\|?message\|?>",
    flags=re.IGNORECASE,
)
_CHANNEL_END_RE = re.compile(r"<\|?end\|?>", flags=re.IGNORECASE)
_MIRRORED_REASONING_WITH_FINAL_RE = re.compile(
    r"^<\|channel>.*?<channel\|>(?P<final>\s*\S[\s\S]*)$",
    flags=re.IGNORECASE | re.DOTALL,
)
_MARKER_RE = re.compile(
    r"(?im)^\s*(final|final answer|answer|output|result|prompt)\s*[:\-]\s*",
)
_IM_TOKEN_RE = re.compile(
    r"(?i)<\|?im_(start|end)\|?>|<im_(start|end)>|<\|endoftext\|>",
)
_PLANNING_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?\s*(?:"
    r"final\s+plan\b|"
    r"final\s+check\b|"
    r"i\s+(?:should|need|must|will|want|am\s+going\s+to|have\s+to)\b|"
    r"let's\b|"
    r"first\s*[,.:]|next\s*[,.:]|then\s*[,.:]|"
    r"wait\s*[,.:]|"
    r"ready\s+to\s+write\b|"
    r"writing\s+the\s+prompt\b|"
    r"so\s+i\s+need\s+to\b|"
    r"i\s+should\s+focus\s+on\b"
    r")"
)


def clean_model_output(text: str, config: OutputCleanConfig | None = None) -> str:
    if not text:
        return ""

    cfg = config or OutputCleanConfig()
    cleaned = (text or "").strip()

    # Remove common chat template tokens that sometimes leak into output.
    cleaned = _IM_TOKEN_RE.sub("", cleaned).strip()

    if cfg.strip_think:
        cleaned = _strip_channel_protocol(cleaned)
        cleaned = _THINK_BLOCK_RE.sub("", cleaned)
        cleaned = _THINK_CLOSE_RE.sub("", cleaned)
        unclosed_think = _THINK_OPEN_RE.search(cleaned)
        if unclosed_think:
            # When generation ends before </think>, no final answer exists.
            # Prompt-shaped examples inside that unfinished reasoning must not
            # be exposed as RESPONSE.
            cleaned = cleaned[:unclosed_think.start()]
        cleaned = cleaned.strip()

    cleaned = _IM_TOKEN_RE.sub("", cleaned).strip()

    if cfg.strip_code_fences and "```" in cleaned:
        lines = [ln for ln in cleaned.splitlines() if not _CODE_FENCE_RE.match(ln)]
        cleaned = "\n".join(lines).strip()

    if cfg.strip_json_wrappers:
        maybe = _extract_from_json(cleaned, mode=cfg.mode)
        if maybe is not None:
            cleaned = maybe.strip()

    if cfg.strip_leading_preamble:
        cleaned = _drop_preamble(cleaned).strip()

    if cfg.strip_planning and cfg.mode == "prompt":
        without_planning = _strip_planning_paragraphs(cleaned)
        if without_planning:
            cleaned = without_planning

    if cfg.strip_role_prefixes:
        lines = cleaned.splitlines()
        if lines:
            lines[0] = _ROLE_PREFIX_RE.sub("", lines[0])
        cleaned = "\n".join(lines).strip()

    cleaned = _MARKER_RE.sub("", cleaned).strip()

    if cfg.keep_first_paragraph_only:
        parts = re.split(r"\n\s*\n", cleaned, maxsplit=1)
        cleaned = parts[0].strip()

    return cleaned


def _strip_channel_protocol(text: str) -> str:
    """Remove reasoning only when the whole output starts as a channel transcript."""
    candidate = (text or "").strip()
    channel_opens = list(_CHANNEL_OPEN_RE.finditer(candidate))
    if channel_opens and channel_opens[0].start() == 0:
        final_segments: list[str] = []
        non_channel_text: list[str] = []

        for index, channel_open in enumerate(channel_opens):
            chunk_end = (
                channel_opens[index + 1].start()
                if index + 1 < len(channel_opens)
                else len(candidate)
            )
            chunk = candidate[channel_open.end():chunk_end]
            channel_end = _CHANNEL_END_RE.search(chunk)
            content = chunk[:channel_end.start()] if channel_end else chunk

            if channel_open.group("channel").lower() == "final":
                final_segments.append(content.strip())
            elif channel_end:
                non_channel_text.append(chunk[channel_end.end():])

        if final_segments:
            return final_segments[-1]
        return "".join(non_channel_text).strip()

    mirrored = _MIRRORED_REASONING_WITH_FINAL_RE.fullmatch(candidate)
    if mirrored:
        return mirrored.group("final").strip()

    return candidate


def _extract_from_json(text: str, mode: str) -> str | None:
    candidate = text.strip()
    if not candidate:
        return None

    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None

    try:
        payload = json.loads(candidate)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    preferred_keys = []
    if mode == "prompt":
        preferred_keys = ["prompt", "final", "output", "text", "content"]
    else:
        preferred_keys = ["text", "content", "output", "final"]

    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return None


def _drop_preamble(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    keep_from = 0
    for i, ln in enumerate(lines[:20]):
        if _MARKER_RE.match(ln):
            keep_from = i
        if re.search(r"(?i)\bhere(?:'s| is)\b", ln) and i < 6:
            keep_from = max(keep_from, i + 1)

    return "\n".join(lines[keep_from:]).strip()


def _strip_planning_paragraphs(text: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]
    if not paragraphs:
        return ""

    kept: list[str] = []
    dropping = True
    for p in paragraphs:
        is_planning = bool(_PLANNING_RE.search(p))
        if dropping and is_planning:
            continue
        dropping = False
        kept.append(p)

    # If everything looked like planning, don't destroy the output.
    if not kept:
        return text.strip()
    return "\n\n".join(kept).strip()


def prompt_output_guard() -> str:
    return (
        "Output rules that override all style preferences: return only the final usable prompt text. "
        "Do not include analysis, planning, bullet points, headings, markdown, JSON, explanations, alternatives, self-corrections, or notes. "
        "Do not write phrases such as Final Plan, Final Check, Wait, Okay, First, Next, Then, I will, or I need. "
        "Start directly with the required prompt text."
    )
