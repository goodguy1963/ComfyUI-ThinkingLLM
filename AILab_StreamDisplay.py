import re
import shutil
import sys
import time

# ── Global singleton for live terminal streaming ──────────────────────────
THINKING_STREAM_DISPLAY = None


def get_thinking_stream_display():
    """Return or create a compact-mode TerminalStreamDisplay for live
    token output during LLM inference.  Use the same label across all
    nodes so the user sees a single coherent stream in the terminal."""
    global THINKING_STREAM_DISPLAY
    if THINKING_STREAM_DISPLAY is None:
        THINKING_STREAM_DISPLAY = TerminalStreamDisplay(
            label="ThinkingLLM",
            flush_chars=160,
            min_chunk_chars=48,
            flush_interval=0.8,
            compact=True,
        )
    return THINKING_STREAM_DISPLAY


def reset_thinking_stream_display():
    """Kill the singleton so the next call to *get_thinking_stream_display*
    creates a fresh display.  Safe to call between queues."""
    global THINKING_STREAM_DISPLAY
    if THINKING_STREAM_DISPLAY is not None:
        THINKING_STREAM_DISPLAY.end_stage()
    THINKING_STREAM_DISPLAY = None


# ── Shared chunk normalizer for OpenAI-style streamed responses ────────────

def _get_stream_field(container, key: str, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _stream_value_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "".join(_stream_value_to_text(part) for part in value)
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning_content", "reasoning", "thinking"):
            text = _stream_value_to_text(value.get(key))
            if text:
                return text
        return ""
    return str(value)


def _first_stream_text(containers, keys: tuple[str, ...]) -> str:
    for container in containers:
        if container is None:
            continue
        for key in keys:
            text = _stream_value_to_text(_get_stream_field(container, key, None))
            if text:
                return text
    return ""


def extract_stream_token(chunk) -> dict[str, str]:
    """Return a dict with 'reasoning' and 'content' keys from an OpenAI-style
    streaming delta chunk.

    Populates both fields from any supported field name so callers get reasoning
    and answer text regardless of backend (llama.cpp, vLLM, HF, etc.).
    Empty values are returned as empty strings.
    """
    choices = _get_stream_field(chunk, "choices", []) or []
    choice = choices[0] if isinstance(choices, (list, tuple)) and choices else {}
    delta = _get_stream_field(choice, "delta", {}) or {}
    message = _get_stream_field(choice, "message", {}) or {}

    containers = (delta, message, choice, chunk)
    content = _first_stream_text(containers, ("content", "text", "output_text"))
    reasoning = _first_stream_text(
        containers,
        ("reasoning_content", "thinking", "reasoning", "reasoning_text"),
    )

    return {"reasoning": reasoning, "content": content}


class StreamDegenerationError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class StreamDegenerationGuard:
    def __init__(self, repeated_word_limit: int = 8, repeated_ngram_limit: int = 8):
        self.repeated_word_limit = max(4, int(repeated_word_limit))
        self.repeated_ngram_limit = max(3, int(repeated_ngram_limit))
        self._text = ""
        self._words: list[str] = []

    def push(self, text: str):
        if not text:
            return
        self._text = (self._text + text)[-4096:]
        words = [w.casefold() for w in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", self._text)]
        if not words:
            return
        self._words = words[-256:]
        reason = self._detect()
        if reason:
            raise StreamDegenerationError(reason)

    def _detect(self) -> str:
        if not self._words:
            return ""
        last_word = self._words[-1]
        if len(last_word) >= 4:
            run = 0
            for word in reversed(self._words):
                if word != last_word:
                    break
                run += 1
            if run >= self.repeated_word_limit:
                return f"repeated word '{last_word}' {run} times"

        for size in range(2, 6):
            needed = size * self.repeated_ngram_limit
            if len(self._words) < needed:
                continue
            tail = self._words[-size:]
            if all(self._words[-offset - size : -offset] == tail for offset in range(size, needed, size)):
                return f"repeated {size}-word phrase {' '.join(tail)!r}"
        return ""


def strip_degenerate_repetition(text: str, repeated_word_limit: int = 8) -> str:
    if not text:
        return ""
    matches = list(re.finditer(r"[A-Za-z][A-Za-z'\-]{2,}", text))
    if not matches:
        return text
    last = matches[-1].group(0).casefold()
    if len(last) < 4:
        return text
    run_start = len(matches) - 1
    while run_start >= 0 and matches[run_start].group(0).casefold() == last:
        run_start -= 1
    run_count = len(matches) - run_start - 1
    if run_count < repeated_word_limit:
        return text
    cut_at = matches[run_start + 1].start()
    return text[:cut_at].rstrip(" \t\r\n,.;:!?")


class TerminalStreamDisplay:
    """Buffered terminal display for streamed model output.

    Keeps the raw output accumulation in the caller, but renders only readable
    chunks to the terminal with stage headers and completion summaries.

    When *compact* is True, token content is printed as clean wrapped text
    without node labels, carriage returns, or tail truncation.
    """

    _SENTENCE_BOUNDARY_RE = re.compile(r"[.!?](?:['\")\]]+)?(?:\s+|$)")
    _PARAGRAPH_BOUNDARY_RE = re.compile(r"\n\s*\n")
    _PLANNING_RE = re.compile(
        r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?\s*(okay[,.:]?|first[,.:]?|next[,.:]?|then[,.:]?|wait[,.:]?|final\s+plan|final\s+check)\b"
    )
    _FIRST_PERSON_RE = re.compile(r"(?i)\b(i\s+(should|need|must|will|am\s+going\s+to|have\s+to))\b")
    TAIL_LENGTH = 200

    def __init__(
        self,
        label: str,
        flush_chars: int = 160,
        min_chunk_chars: int = 48,
        flush_interval: float = 1.0,
        suppress_planning: bool = False,
        compact: bool = True,
        line_width: int | None = None,
    ):
        self.label = label
        self.flush_chars = max(32, int(flush_chars))
        self.min_chunk_chars = max(1, int(min_chunk_chars))
        self.flush_interval = max(0.1, float(flush_interval))
        self.suppress_planning = suppress_planning
        self.compact = compact
        self._buffer = ""
        self._last_flush_at = time.monotonic()
        self._stage_name = None
        self._stage_started_at = None
        self._stage_chars = 0
        self._stage_updates = 0
        self._last_compact_emit_at = 0.0
        self._last_compact_tail = ""
        self._compact_active = False
        self._stream_pending_text = ""
        self._stream_col = 0
        self._stream_need_space = False
        self._line_width_override = int(line_width) if line_width else None

    def start_stage(self, stage_name: str):
        self.end_stage()
        self._stage_name = stage_name
        self._stage_started_at = time.monotonic()
        self._stage_chars = 0
        self._stage_updates = 0
        if self.compact:
            return
        self._write_line(f"[{self.label}] {stage_name}")

    def push(self, text: str):
        if not text:
            return
        self._stage_chars += len(text)
        self._buffer += text
        if self._should_flush():
            self.flush()

    def flush(self, force: bool = False):
        while True:
            flush_at = self._find_flush_index(force=force)
            if flush_at <= 0:
                break
            chunk = self._buffer[:flush_at]
            self._buffer = self._buffer[flush_at:]
            display_text = self._normalize_chunk(chunk)
            self._last_flush_at = time.monotonic()
            if display_text:
                self._stage_updates += 1
                self._write_line(display_text)
            if not force:
                break

    def _compact_tail(self) -> str:
        """Return a single line containing the rolling last ~200 chars."""
        raw = (self._buffer or "")
        if len(raw) > self.TAIL_LENGTH:
            raw = "…" + raw[-self.TAIL_LENGTH:]
        return re.sub(r"[ \t]+", " ", raw.replace("\r", "").replace("\n", " ")).strip()

    def end_stage(self):
        if self._compact_active:
            self.end_compact()
        else:
            self.flush(force=True)
        if not self._stage_name:
            return
        if self.compact:
            self._stage_name = None
            self._stage_started_at = None
            self._stage_chars = 0
            self._stage_updates = 0
            return
        elapsed = 0.0
        if self._stage_started_at is not None:
            elapsed = max(0.0, time.monotonic() - self._stage_started_at)
        self._write_line(
            f"[{self.label}] {self._stage_name} complete ({elapsed:.1f}s, {self._stage_chars} chars, {self._stage_updates} updates)"
        )
        self._stage_name = None
        self._stage_started_at = None
        self._stage_chars = 0
        self._stage_updates = 0

    def _should_flush(self) -> bool:
        if not self._buffer:
            return False
        if len(self._buffer) >= self.flush_chars:
            return True
        if self._last_boundary_index() >= self.min_chunk_chars:
            return True
        if (time.monotonic() - self._last_flush_at) >= self.flush_interval and len(self._buffer) >= self.min_chunk_chars:
            return True
        return False

    def _find_flush_index(self, force: bool) -> int:
        if not self._buffer:
            return 0
        if force:
            return len(self._buffer)
        boundary_index = self._last_boundary_index()
        if boundary_index >= self.min_chunk_chars:
            return boundary_index
        if len(self._buffer) >= self.flush_chars:
            whitespace_index = self._buffer.rfind(" ", 0, self.flush_chars)
            return whitespace_index + 1 if whitespace_index > 0 else self.flush_chars
        if (time.monotonic() - self._last_flush_at) >= self.flush_interval and len(self._buffer) >= self.min_chunk_chars:
            whitespace_index = self._buffer.rfind(" ")
            return whitespace_index + 1 if whitespace_index > 0 else len(self._buffer)
        return 0

    def _last_boundary_index(self) -> int:
        last_end = 0
        for match in self._PARAGRAPH_BOUNDARY_RE.finditer(self._buffer):
            last_end = max(last_end, match.end())
        for match in self._SENTENCE_BOUNDARY_RE.finditer(self._buffer):
            last_end = max(last_end, match.end())
        return last_end

    def _normalize_chunk(self, chunk: str) -> str:
        text = chunk.replace("\r", "")
        parts = []
        for block in re.split(r"\n\s*\n", text):
            normalized = re.sub(r"[ \t]+", " ", block).strip()
            if not normalized:
                continue
            if self.suppress_planning and self._looks_like_planning(normalized):
                continue
            parts.append(normalized)
        return "\n".join(parts)

    def _looks_like_planning(self, text: str) -> bool:
        return bool(self._PLANNING_RE.search(text) or self._FIRST_PERSON_RE.search(text))

    def _write_line(self, text: str):
        if not text:
            return
        msg = f"{text}\n"
        sys.stdout.write(msg)
        sys.stdout.flush()

    def _stream_line_width(self) -> int:
        if self._line_width_override:
            return max(24, self._line_width_override)
        columns = shutil.get_terminal_size(fallback=(120, 24)).columns
        return max(40, min(96, columns // 2))

    def _write_wrapped_stream_text(self, text: str):
        if not text:
            return
        width = self._stream_line_width()
        for part in re.findall(r"\S+|\s+", text.replace("\r", "")):
            if not part:
                continue
            if part.isspace():
                newline_count = part.count("\n")
                if newline_count:
                    sys.stdout.write("\n" * newline_count)
                    self._stream_col = 0
                    self._stream_need_space = False
                elif self._stream_col > 0:
                    self._stream_need_space = True
                continue

            word = part
            prefix = 1 if self._stream_need_space and self._stream_col > 0 else 0
            if self._stream_col > 0 and self._stream_col + prefix + len(word) > width:
                sys.stdout.write("\n")
                self._stream_col = 0
                prefix = 0
            if prefix:
                sys.stdout.write(" ")
                self._stream_col += 1
            sys.stdout.write(word)
            self._stream_col += len(word)
            self._stream_need_space = False
        sys.stdout.flush()

    def _drain_stream_pending(self, force: bool = False):
        if not self._stream_pending_text:
            return
        text = self._stream_pending_text
        if force:
            emit = text
            self._stream_pending_text = ""
        else:
            match = None
            for match in re.finditer(r"\s+", text):
                pass
            if match is None:
                if len(text) < self._stream_line_width():
                    return
                emit = text
                self._stream_pending_text = ""
            else:
                end = match.end()
                emit = text[:end]
                self._stream_pending_text = text[end:]
        self._write_wrapped_stream_text(emit)

    def push_compact(self, text: str):
        """Append *text* as clean wrapped terminal output without node prefixes."""
        if not text:
            return
        self._compact_active = True
        self._stage_chars += len(text)
        self._stage_updates += 1
        self._stream_pending_text += text
        self._drain_stream_pending(force=False)

    def end_compact(self):
        """Finish compact mode with one final newline-terminated snapshot."""
        had_pending = bool(self._stream_pending_text)
        self._drain_stream_pending(force=True)
        if (had_pending or self._stream_col > 0) and self._stage_chars:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._last_compact_tail = ""
        self._last_compact_emit_at = 0.0
        self._compact_active = False
        self._buffer = ""
        self._stream_pending_text = ""
        self._stream_col = 0
        self._stream_need_space = False
