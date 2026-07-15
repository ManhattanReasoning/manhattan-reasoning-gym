from __future__ import annotations

import os
from collections.abc import Callable

from . import _client, _credentials, _progress

# All App instances created at module import time are registered here so the
# CLI can discover them without the user needing to export anything explicitly.
_registry: list[App] = []


class RegisterMap:
    """Base class for FPGA register address maps.

    Subclass and define integer class attributes for named register byte addresses:

        class Regs(manhattan_reasoning_gym.RegisterMap):
            CTRL     = 0x0000
            DATA_IN  = 0x0004
            DATA_OUT = 0x0008
    """


class App:
    """A Manhattan Reasoning Gym application: design source + register map + API config.

    The API key is resolved automatically (explicit ``api_key`` arg >
    ``$MRG_API_KEY`` > the key stored by ``mrg login``), so it
    usually doesn't need to be passed explicitly.

    ``fpga_id`` is optional and normally left unset -- a fresh build never
    picks a board itself, the server assigns whichever one frees up first once
    the build finishes, and this fills in automatically once ``_program()``
    completes. Pass it explicitly only to reconnect to a board you already
    have a live session on without rebuilding (``mrg run --no-program
    --fpga-id N`` / ``App(..., fpga_id=N)`` with programming skipped).

    Typical usage::

        app = manhattan_reasoning_gym.App(
            "my_design",
            design="path/to/design.py",
        )

        @app.local_entrypoint()
        def main():
            app.write(Regs.DATA_IN, 0xDEADBEEF)
            print(hex(app.read(Regs.DATA_OUT)))

    Run with::

        mrg run myfile.py
    """

    def __init__(
        self,
        name: str,
        *,
        design: str,
        fpga_id: int | None = None,
        registers: type[RegisterMap] | None = None,
        api_key: str | None = None,
        api_url: str = _client.DEFAULT_API_URL,
        sys_clk_freq: int | None = None,
        timing_target_mhz: float | None = None,
    ) -> None:
        self.name = name
        self.design = design
        self.fpga_id = fpga_id
        self.registers = registers
        self.api_url = api_url
        # Optional override for the SoC compute-domain clock (Hz). None lets the
        # build server use its default (50 MHz). Falls back to $MRG_SYS_CLK_FREQ
        # so the same knob works from the CLI, the env, or an explicit arg.
        _env_clk = os.environ.get("MRG_SYS_CLK_FREQ")
        self.sys_clk_freq = sys_clk_freq or (int(_env_clk) if _env_clk else None)
        # Optional PnR/grading timing target (MHz); defaults server-side to the
        # sys clock. Falls back to $MRG_TIMING_TARGET_MHZ, mirroring sys_clk_freq.
        _env_tt = os.environ.get("MRG_TIMING_TARGET_MHZ")
        self.timing_target_mhz = timing_target_mhz or (
            float(_env_tt) if _env_tt else None
        )
        # Resolve the key: explicit arg > $MRG_API_KEY > stored login.
        # Left empty if none found; the CLI re-checks before running, and
        # programmatic callers can set .api_key later.
        self.api_key = (
            api_key
            or os.environ.get("MRG_API_KEY")
            or _credentials.load(api_url)
            or ""
        )
        self._entrypoint: Callable | None = None
        self._programmed = False
        _registry.append(self)

    def local_entrypoint(self) -> Callable:
        """Decorator that marks a function as the CLI entrypoint for this app.

        The decorated function is called by ``mrg run`` after the FPGA
        has been programmed.  It is NOT called automatically when the file is
        imported or run with ``python``.
        """
        def decorator(fn: Callable) -> Callable:
            self._entrypoint = fn
            return fn
        return decorator

    def read(self, addr: int, count: int = 1) -> int | list[int]:
        """Read ``count`` 32-bit words starting at byte address ``addr``.

        Returns a single ``int`` when *count* is 1, or a ``list[int]`` otherwise.
        Programs the FPGA first if not already done.
        """
        self._ensure_programmed()
        data = _client.read(self.fpga_id, self.api_key, addr, count, self.api_url)
        return data[0] if count == 1 else data

    def write(
        self, addr: int, value: int | list[int], fixed_address: bool = False
    ) -> None:
        """Write one or more 32-bit words to byte address ``addr``.

        *value* may be a single ``int`` or a ``list[int]`` for a burst write.
        ``fixed_address=True`` repeats ``addr`` for every word instead of
        incrementing it -- for a hardware FIFO/push-register port (a design
        that keeps its own internal write_idx), where a normal burst would
        scatter words across whatever registers happen to sit at
        address+1, address+2, ... Programs the FPGA first if not already done.
        """
        self._ensure_programmed()
        words = [value] if isinstance(value, int) else value
        _client.write(
            self.fpga_id, self.api_key, addr, words, self.api_url,
            fixed_address=fixed_address,
        )

    def stream(self) -> _client.Stream:
        """Open a persistent, low-latency session for many small ops.

        Unlike ``write()``/``read()``, ops on the returned ``Stream`` skip the
        per-call job queue and poll loop -- use this for tight loops like
        streaming a CNF instance in one literal per write, or an RL reward
        loop that needs many ops per episode. Programs the FPGA first if not
        already done.
        """
        self._ensure_programmed()
        return _client.Stream(self.fpga_id, self.api_key, self.api_url)

    def _program(self) -> None:
        """Submit the design and block until it's built and flashed.

        No board is chosen up front: the server assigns one once the build
        finishes and some board's worker is ready to flash it. ``self.fpga_id``
        is set from the completed job, not decided here -- it's discovered,
        not picked.
        """
        job_id = _client.submit(
            self.design, self.api_key, self.api_url,
            sys_clk_freq=self.sys_clk_freq,
            timing_target_mhz=self.timing_target_mhz,
        )
        bar = _progress.BuildProgress(self.name)
        try:
            job = _client.poll_job(
                job_id, self.api_key, self.api_url, on_poll=bar.update
            )
        except BaseException:
            bar.abort()
            raise
        self.fpga_id = job["fpga_id"]
        bar.finish(self.fpga_id)
        self._programmed = True

    def release(self) -> str:
        """Release the active session, returning the reset job_id."""
        job_id = _client.release_session(self.fpga_id, self.api_key, self.api_url)
        self._programmed = False
        return job_id

    def __enter__(self) -> App:
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def _ensure_programmed(self) -> None:
        if not self._programmed:
            self._program()
