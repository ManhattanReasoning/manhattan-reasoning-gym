from __future__ import annotations

import os
import signal
import threading
from collections.abc import Callable

import requests

from . import _client, _credentials, _progress, _transport

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

    ``read()``/``write()`` go over one WebSocket, opened on the first call and
    closed after a short idle period (the next call reopens it). Pass
    ``transport="http"`` -- or set ``$MRG_TRANSPORT=http`` -- to make each call
    its own HTTP job instead: much slower, but it works on networks that block
    WebSockets.

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
        sources: list[str] | None = None,
        include_dir: str | None = None,
        top: str | None = None,
        fpga_id: int | None = None,
        registers: type[RegisterMap] | None = None,
        api_key: str | None = None,
        api_url: str = _client.DEFAULT_API_URL,
        sys_clk_freq: int | None = None,
        timing_target_mhz: float | None = None,
        poll_timeout: float | None = None,
        transport: str | None = None,
    ) -> None:
        self.name = name
        self.design = design
        self.sources = None if sources is None else list(sources)
        self.include_dir = include_dir
        # How long _program() waits for a build before giving up. None takes
        # _client.DEFAULT_POLL_TIMEOUT, which is sized to outlast the server's
        # own build ceiling. Raise it only if your designs build slower than
        # that; lowering it risks abandoning builds that would have succeeded.
        self.poll_timeout = poll_timeout
        # How read()/write() reach the board: "websocket" (default) keeps one
        # socket open across calls; "http" makes each call its own queued job
        # -- much slower, for networks that block WebSockets. Falls back to
        # $MRG_TRANSPORT; `mrg run --transport` overrides it.
        self.transport = _transport.resolve_transport(transport)
        # Verilog-only top-module disambiguator (ignored for an Amaranth
        # design); only needed when the file has more than one module
        # exposing the Wishbone contract -- otherwise auto-detected.
        self.top = top
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
            api_key or os.environ.get("MRG_API_KEY") or _credentials.load(api_url) or ""
        )
        self._entrypoint: Callable | None = None
        self._programmed = False
        self._connection: _transport.Transport | None = None
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
        data = self._open_connection().read(addr, count)
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
        self._open_connection().write(addr, words, fixed_address=fixed_address)

    def _program(self) -> None:
        """Submit the design and block until it's built and flashed.

        No board is chosen up front: the server assigns one once the build
        finishes and some board's worker is ready to flash it. ``self.fpga_id``
        is set from the completed job, not decided here -- it's discovered,
        not picked.
        """
        job_id = _client.submit(
            self.design,
            self.api_key,
            self.api_url,
            top=self.top,
            sys_clk_freq=self.sys_clk_freq,
            timing_target_mhz=self.timing_target_mhz,
            **({"sources": self.sources} if self.sources is not None else {}),
            **(
                {"include_dir": self.include_dir}
                if self.include_dir is not None else {}
            ),
        )
        bar = _progress.BuildProgress(self.name)
        try:
            job = _client.poll_job(
                job_id,
                self.api_key,
                self.api_url,
                timeout=self.poll_timeout,
                on_poll=bar.update,
            )
        except BaseException:
            bar.abort()
            try:
                _client.cancel_job(job_id, self.api_key, self.api_url)
            except Exception:
                pass
            raise
        self.fpga_id = job["fpga_id"]
        bar.finish(self.fpga_id)
        self._programmed = True

    def release(self) -> str:
        """Release the active session, returning the reset job_id.

        Closes this App's WebSocket first: an open socket holds the board's
        link lock, which the reset job needs.
        """
        if self.fpga_id is None:
            return ""

        self._close_connection()

        orig_handler = None
        try:
            # Shield the network call from SIGINT so mashing Ctrl-C does not abort
            # the release request while it's in flight.
            if threading.current_thread() is threading.main_thread():
                try:
                    orig_handler = signal.getsignal(signal.SIGINT)
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                except (ValueError, TypeError):
                    orig_handler = None
            try:
                job_id = _client.release_session(
                    self.fpga_id, self.api_key, self.api_url
                )
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    # Session key expired or missing in Redis while the board is still
                    # reserved; fall back to direct reset to return the FPGA to idle.
                    res = _client.reset_fpga(self.fpga_id, self.api_key, self.api_url)
                    job_id = res.get("job_id", "")
                else:
                    raise
            self._programmed = False
            return job_id
        finally:
            if orig_handler is not None:
                signal.signal(signal.SIGINT, orig_handler)

    def __enter__(self) -> App:
        return self

    def __exit__(self, exc_type, *_) -> None:
        try:
            self.release()
        except Exception:
            # Do not let release errors mask exceptions raised inside the with-block
            if exc_type is None:
                raise

    def _ensure_programmed(self) -> None:
        if not self._programmed:
            self._program()

    def _open_connection(self) -> _transport.Transport:
        # Made on first use, not in __init__: fpga_id is only known once the
        # board has been programmed.
        if self._connection is None:
            self._connection = _transport.open_transport(
                self.transport, self.fpga_id, self.api_key, self.api_url
            )
        return self._connection

    def _close_connection(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
