import re
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


class TerminalStreamDisplay:
    """Buffered terminal display for streamed model output.

    Keeps the raw output accumulation in the caller, but renders only readable
    chunks to the terminal with stage headers and completion summaries.

    When *compact* is True, token content is shown as a rolling 200-character
    tail on a single line so older tokens scroll off naturally.
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

    def start_stage(self, stage_name: str):
        self.end_stage()
        self._stage_name = stage_name
        self._stage_started_at = time.monotonic()
        self._stage_chars = 0
        self._stage_updates = 0
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
        self.flush(force=True)
        if not self._stage_name:
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

    def _write_rolling_line(self, label: str, tail: str, final: bool = False):
        """Emit a carriage-return line showing the last ~200 chars, clearing to EOL.
        'final' adds a trailing newline so the last frame stays visible."""
        if not tail:
            return
        msg = f"\r[{label}] {tail}\033[K"
        sys.stdout.write(msg)
        if final:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def push_compact(self, text: str):
        """Append *text* and emit a carriage-return rolling tail every throttle interval.

        Uses \\r to redraw a single line so the terminal stays clean without
        flooding the ComfyUI log buffer.
        """
        if not text:
            return
        self._stage_chars += len(text)
        self._buffer += text
        if len(self._buffer) > self.TAIL_LENGTH * 2:
            self._buffer = self._buffer[-self.TAIL_LENGTH:]
        tail = self._compact_tail()
        now = time.monotonic()
        should_emit = False
        if tail and tail != self._last_compact_tail:
            if self._last_compact_emit_at <= 0.0:
                should_emit = len(self._buffer) >= self.min_chunk_chars
            elif (now - self._last_compact_emit_at) >= self.flush_interval:
                should_emit = True
        if should_emit:
            self._stage_updates += 1
            self._last_compact_tail = tail
            self._last_compact_emit_at = now
            self._write_rolling_line(self.label, tail)

    def end_compact(self):
        """Finish compact mode with one final newline-terminated snapshot."""
        tail = self._compact_tail()
        if tail and tail != self._last_compact_tail:
            self._stage_updates += 1
        self._last_compact_tail = ""
        self._last_compact_emit_at = 0.0
        self._write_rolling_line(self.label, tail or "", final=True)
