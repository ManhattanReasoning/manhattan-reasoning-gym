"""Live terminal progress for long-running builds — dependency-free ANSI.

Renders a single self-updating line: a braille spinner, the real job phase
(queued/building/flashing), a rotating silly quip, and an mm:ss stopwatch.
Falls back to plain phase-change lines when stdout isn't a TTY (pipes, CI).
"""

from __future__ import annotations

import sys
import time
from typing import TextIO

# ── ANSI ─────────────────────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CLEAR_LINE = "\r\033[K"

# ── animation tuning ─────────────────────────────────────────────────────────
_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_FPS = 12.5            # spinner frame advances every ~0.08s
_QUIP_PERIOD = 3.0    # seconds each quip stays up
_SHIMMER_SPEED = 7.0  # rainbow colour steps per second on the quip

# A smooth loop around the 256-colour cube (red→orange→yellow→green→cyan→blue
# →magenta→red). The quip text slowly cycles through these for a silly shimmer.
_RAINBOW = (
    196, 202, 208, 214, 220, 226, 190, 154, 118, 82,
    46, 47, 48, 49, 50, 51, 45, 39, 33, 27,
    21, 57, 93, 129, 165, 201, 200, 199, 198, 197,
)


def _rainbow_code(step: int) -> int:
    """256-colour code for the given shimmer step (wraps around the loop)."""
    return _RAINBOW[step % len(_RAINBOW)]

_QUIPS = (
    "bribing the place-and-route gods",
    "convincing electrons to cooperate",
    "routing nets through lower Manhattan",
    "negotiating with timing constraints",
    "teaching sand to think",
    "aligning the lookup tables",
    "untangling combinational logic",
    "warming up the flip-flops",
    "consulting the timing oracle",
    "reticulating splines",
)


def _fmt_clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


class BuildProgress:
    """Drives the live build line. Call ``update(status, fpga_id)`` often;
    then ``finish(fpga_id)`` on success or ``abort()`` before raising on
    failure.

    No board is known at construction time -- a fresh build doesn't pick one,
    the server assigns whichever frees up first once the build finishes (see
    ``_app.App._program``) -- so ``fpga_id`` starts unset and is filled in
    from whatever ``update``/``finish`` are called with.
    """

    def __init__(self, name: str, stream: TextIO | None = None) -> None:
        self.name = name
        self.fpga_id: int | None = None
        self.stream = stream or sys.stdout
        self.color = self.stream.isatty()
        self.start = time.monotonic()
        self._last_status: str | None = None

    def _c(self, text: str, *codes: str) -> str:
        if not self.color:
            return text
        return "".join(codes) + text + _RESET

    def _phase(self, status: str) -> str:
        """Human phase label for a raw job status, given what's known so far.

        ``running`` covers both the Fargate build and the flash -- the two are
        only distinguishable by whether a board has been assigned yet, since
        that's the moment board assignment actually happens (see CLAUDE.md,
        "Planned: decouple build concurrency from board count" in the
        orchestrator repo).
        """
        if status == "running":
            return (
                f"flashing fpga{self.fpga_id}"
                if self.fpga_id is not None
                else "building bitstream"
            )
        return status

    def _shimmer(self, text: str, elapsed: float) -> str:
        """Wrap text in a slowly-cycling 256-colour code for a silly shimmer."""
        code = _rainbow_code(int(elapsed * _SHIMMER_SPEED))
        return f"\033[1m\033[38;5;{code}m{text}{_RESET}"

    def update(self, status: str, fpga_id: int | None = None) -> None:
        if fpga_id is not None:
            self.fpga_id = fpga_id
        elapsed = time.monotonic() - self.start
        if not self.color:
            # Non-TTY: emit one line per phase change, no animation.
            if status != self._last_status:
                self._last_status = status
                self.stream.write(f"[mrg] {self._phase(status)}...\n")
                self.stream.flush()
            return

        frame = _FRAMES[int(elapsed * _FPS) % len(_FRAMES)]
        quip = _QUIPS[int(elapsed / _QUIP_PERIOD) % len(_QUIPS)]
        line = (
            f"{_CLEAR_LINE}"
            f"{self._c(frame, _CYAN, _BOLD)} "
            f"{self._c(self._phase(status), _BOLD)} "
            f"{self._c(f'· {self.name} · ', _DIM)}"
            f"{self._shimmer(quip + '…', elapsed)} "
            f"{self._c(_fmt_clock(elapsed), _YELLOW)}"
        )
        self.stream.write(line)
        self.stream.flush()

    def finish(self, fpga_id: int | None = None) -> None:
        if fpga_id is not None:
            self.fpga_id = fpga_id
        elapsed = time.monotonic() - self.start
        if self.color:
            self.stream.write(_CLEAR_LINE)
        mark = self._c("✓", _GREEN, _BOLD)
        tail = f"· {self.name} on fpga{self.fpga_id} · {elapsed:.1f}s"
        self.stream.write(f"{mark} {self._c('ready', _BOLD)} {self._c(tail, _DIM)}\n")
        self.stream.flush()

    def abort(self) -> None:
        """Clear the live line so a following error/traceback prints cleanly."""
        if self.color:
            self.stream.write(_CLEAR_LINE)
            self.stream.flush()


def note(msg: str, stream: TextIO | None = None) -> None:
    """Print a dim one-off status note in the same house style."""
    stream = stream or sys.stdout
    text = f"\033[2m[mrg] {msg}\033[0m" if stream.isatty() else f"[mrg] {msg}"
    print(text, file=stream)
