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

    ``fpga_id`` is optional: leave it unset (or pass ``None``) to let the SDK
    pick an idle FPGA at program time, or pin a specific board by passing an id.

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

    def write(self, addr: int, value: int | list[int]) -> None:
        """Write one or more 32-bit words to byte address ``addr``.

        *value* may be a single ``int`` or a ``list[int]`` for a burst write.
        Programs the FPGA first if not already done.
        """
        self._ensure_programmed()
        words = [value] if isinstance(value, int) else value
        _client.write(self.fpga_id, self.api_key, addr, words, self.api_url)

    def _resolve_fpga(self) -> None:
        """Pick an idle FPGA if one wasn't pinned. Caches the choice on self."""
        if self.fpga_id is None:
            self.fpga_id = _client.find_idle_fpga(self.api_key, self.api_url)
            _progress.note(f"picked idle fpga{self.fpga_id}")

    def _program(self) -> None:
        self._resolve_fpga()
        job_id = _client.submit(
            self.fpga_id, self.design, self.api_key, self.api_url,
            sys_clk_freq=self.sys_clk_freq,
            timing_target_mhz=self.timing_target_mhz,
        )
        bar = _progress.BuildProgress(self.name, self.fpga_id)
        try:
            _client.poll_job(
                self.fpga_id, job_id, self.api_key, self.api_url, on_poll=bar.update
            )
        except BaseException:
            bar.abort()
            raise
        bar.finish()
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
