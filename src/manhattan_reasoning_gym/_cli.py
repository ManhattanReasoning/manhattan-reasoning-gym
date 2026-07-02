"""mrg CLI entry point."""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import sys
from pathlib import Path

from . import _client, _credentials

# ── terminal colour helpers ──────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[31m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"

_USE_COLOR = sys.stdout.isatty()


def _c(text: str, *codes: str) -> str:
    if not _USE_COLOR:
        return text
    return "".join(codes) + text + _RESET


_STATE_COLOR = {
    "idle":         _GREEN,
    "queued":       _YELLOW,
    "building":     _YELLOW,
    "programming":  _YELLOW,
    "reserved":     _CYAN,
    "error":        _RED,
}

_JOB_COLOR = {
    "queued":    _YELLOW,
    "running":   _YELLOW,
    "complete":  _GREEN,
    "failed":    _RED,
    "cancelled": _DIM,
}


def _fmt_state(s: str) -> str:
    return _c(s, _BOLD, _STATE_COLOR.get(s, ""))


def _fmt_status(s: str) -> str:
    return _c(s, _JOB_COLOR.get(s, ""))


def _fmt_id(s: str | None) -> str:
    if s is None:
        return _c("-", _DIM)
    # Show first 8 chars of UUID so rows stay narrow
    return _c(s[:8] + "…", _DIM)


# ── helpers ──────────────────────────────────────────────────────────────────

def _creds(args: argparse.Namespace) -> tuple[str, str]:
    api_url = args.api_url or _client.DEFAULT_API_URL
    # Precedence: explicit flag > env var > stored login.
    api_key = (
        args.api_key
        or os.environ.get("MRG_API_KEY")
        or _credentials.load(api_url)
        or ""
    )
    if not api_key:
        sys.exit(
            "error: no API key found.\n"
            "  run 'mrg login', set $MRG_API_KEY, or pass --api-key"
        )
    return api_key, api_url


def _load_user_module(path: str):
    p = Path(path).resolve()
    if not p.exists():
        sys.exit(f"error: file not found: {path}")
    spec = importlib.util.spec_from_file_location("_mrg_user", p)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p.parent))
    spec.loader.exec_module(mod)
    return mod


# ── subcommand handlers ──────────────────────────────────────────────────────

def cmd_login(args: argparse.Namespace) -> None:
    import requests

    from . import _oauth

    api_url = args.api_url or _client.DEFAULT_API_URL

    # Token source: explicit PAT > GitHub OAuth device flow (browser) > PAT paste.
    github_token = args.github_token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        client_id = (
            args.client_id
            or os.environ.get("MRG_GITHUB_CLIENT_ID")
            or _oauth.DEFAULT_CLIENT_ID
        )
        if client_id:
            try:
                github_token = _oauth.device_flow_token(
                    client_id,
                    on_prompt=lambda m: print(f"\n  {_c(m, _BOLD)}\n", flush=True),
                )
            except (_oauth.DeviceFlowError, requests.HTTPError) as e:
                sys.exit(f"error: GitHub device login failed: {e}")
        else:
            print(
                "Paste a GitHub token (a personal access token with no scopes is\n"
                "sufficient — it's only used to read your username). Input is hidden.\n"
                "Create one at: https://github.com/settings/tokens"
            )
            github_token = getpass.getpass("GitHub token: ").strip()
    if not github_token:
        sys.exit("error: no GitHub token provided")

    try:
        result = _client.exchange_github_token(github_token, api_url)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        detail = ""
        try:
            detail = e.response.json().get("detail", {}).get("message", "")
        except Exception:
            pass
        sys.exit(f"error: token exchange failed ({status}). {detail}".rstrip())

    path = _credentials.save(api_url, result["api_key"], result["github_username"])
    print(
        f"{_c('✓', _GREEN)} logged in as {_c(result['github_username'], _BOLD)} "
        f"@ {api_url}\n  credentials saved to {path}"
    )


def cmd_logout(args: argparse.Namespace) -> None:
    api_url = args.api_url or _client.DEFAULT_API_URL
    api_key = _credentials.load(api_url)
    if not api_key:
        print(f"not logged in to {api_url}")
        return

    # Best-effort server-side revocation; clear the local key regardless.
    try:
        _client.revoke_key(api_key, api_url)
    except Exception as e:
        print(_c(f"warning: server-side revoke failed: {e}", _YELLOW))

    _credentials.clear(api_url)
    print(f"{_c('✓', _GREEN)} logged out of {api_url}")


def cmd_run(args: argparse.Namespace) -> None:
    from . import _app

    _app._registry.clear()
    _load_user_module(args.file)

    if not _app._registry:
        sys.exit("error: no manhattan_reasoning_gym.App found in file")

    app = _app._registry[-1]

    if args.fpga_id is not None:
        app.fpga_id = args.fpga_id
    if args.sys_clk_freq is not None:
        app.sys_clk_freq = args.sys_clk_freq
    if args.api_url:
        app.api_url = args.api_url
    if args.api_key:
        app.api_key = args.api_key
    elif not app.api_key:
        # Fall back to a stored login for the app's orchestrator URL.
        app.api_key = _credentials.load(app.api_url) or ""
    if not app.api_key:
        sys.exit(
            "error: no API key found.\n"
            "  run 'mrg login', set $MRG_API_KEY, or pass --api-key"
        )

    if not app._entrypoint:
        sys.exit("error: no @app.local_entrypoint() found in file")

    if not args.no_program:
        app._program()

    app._entrypoint()


def cmd_status(args: argparse.Namespace) -> None:
    api_key, api_url = _creds(args)

    if args.fpga_id is not None:
        # Single FPGA detail view
        d = _client.get_fpga(args.fpga_id, api_key, api_url)
        if args.json:
            print(json.dumps(d, indent=2))
            return
        sess = d.get("session") or {}
        print(f"\n  FPGA {d['fpga_id']}")
        print(f"  {'state:':<14} {_fmt_state(d['state'])}")
        if sess:
            print(f"  {'session:':<14} {sess.get('session_id', '-')}")
            print(f"  {'owner:':<14} {sess.get('owner', '-')}")
            print(f"  {'expires at:':<14} {sess.get('expires_at', '-')}")
        else:
            print(f"  {'session:':<14} {_c('-', _DIM)}")
        cj = d.get("current_job_id")
        print(f"  {'current job:':<14} {_c(cj, _DIM) if cj else _c('-', _DIM)}")
        print()
    else:
        # All-FPGAs table
        fpgas = _client.list_fpgas(api_key, api_url)
        if args.json:
            print(json.dumps(fpgas, indent=2))
            return
        header = f"  {'ID':>2}  {'STATE':<14}  {'OWNER':<12}  CURRENT JOB"
        sep    = "  " + "─" * (len(header) - 2)
        print()
        print(_c(header, _BOLD))
        print(_c(sep, _DIM))
        for d in fpgas:
            sess  = d.get("session") or {}
            owner = sess.get("owner", "-") if sess else "-"
            cj    = d.get("current_job_id")
            print(
                f"  {d['fpga_id']:>2}  "
                f"{_fmt_state(d['state']):<14}  "
                f"{owner:<12}  "
                f"{_fmt_id(cj)}"
            )
        print()


def cmd_job(args: argparse.Namespace) -> None:
    api_key, api_url = _creds(args)
    d = _client.get_job(args.fpga_id, args.job_id, api_key, api_url)
    if args.json:
        print(json.dumps(d, indent=2))
        return
    print(f"\n  {'job_id:':<14} {d['job_id']}")
    print(f"  {'fpga_id:':<14} {d['fpga_id']}")
    print(f"  {'type:':<14} {d['type']}")
    print(f"  {'status:':<14} {_fmt_status(d['status'])}")
    print(f"  {'created at:':<14} {d['created_at']}")
    print(f"  {'updated at:':<14} {d['updated_at']}")
    print()


def cmd_logs(args: argparse.Namespace) -> None:
    api_key, api_url = _creds(args)
    text = _client.get_logs(args.fpga_id, args.job_id, api_key, api_url)
    print(text)


def cmd_cancel(args: argparse.Namespace) -> None:
    api_key, api_url = _creds(args)
    _client.cancel_job(args.fpga_id, args.job_id, api_key, api_url)
    print(f"cancelled {args.job_id}")


def cmd_reset(args: argparse.Namespace) -> None:
    api_key, api_url = _creds(args)
    d = _client.reset_fpga(args.fpga_id, api_key, api_url)
    print(f"reset job queued: {d['job_id']}")


def cmd_read(args: argparse.Namespace) -> None:
    api_key, api_url = _creds(args)
    data = _client.read(args.fpga_id, api_key, args.address, args.count, api_url)
    for i, word in enumerate(data):
        print(f"  {hex(args.address + i * 4)}: {word:#010x}")


def cmd_write(args: argparse.Namespace) -> None:
    api_key, api_url = _creds(args)
    _client.write(args.fpga_id, api_key, args.address, [args.value], api_url)
    print(f"  {hex(args.address)} ← {args.value:#010x}")


def _local_report(mode: str, args: argparse.Namespace) -> None:
    """Run a local synth/pnr build and print the JSON report on stdout."""
    from . import _local_build

    try:
        if mode == "synth":
            rep = _local_build.synth(args.design)
        else:
            rep = _local_build.pnr(args.design, target_mhz=args.target_mhz)
    except _local_build.SandboxUnavailableError as exc:
        sys.exit(f"error: {exc}")
    print(rep.to_json())
    sys.exit(0 if rep.ok else 1)


def cmd_synth(args: argparse.Namespace) -> None:
    _local_report("synth", args)


def cmd_pnr(args: argparse.Namespace) -> None:
    _local_report("pnr", args)


# ── parser ───────────────────────────────────────────────────────────────────

def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--api-key", default=None, metavar="KEY",
                        help="API key (default: $MRG_API_KEY)")
    common.add_argument("--api-url", default=None, metavar="URL",
                        help=f"orchestrator URL (default: {_client.DEFAULT_API_URL})")

    parser = argparse.ArgumentParser(
        prog="mrg",
        description="Manhattan Reasoning Gym — program and control FPGAs from the CLI.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # login
    login_p = sub.add_parser(
        "login", parents=[common],
        help="exchange a GitHub token for an API key and store it",
    )
    login_p.add_argument(
        "--github-token", default=None, dest="github_token", metavar="TOKEN",
        help="use this GitHub PAT instead of the browser flow (or $GITHUB_TOKEN)",
    )
    login_p.add_argument(
        "--client-id", default=None, dest="client_id", metavar="ID",
        help="GitHub OAuth App client id for device flow "
             "(default: $MRG_GITHUB_CLIENT_ID)",
    )

    # logout
    sub.add_parser(
        "logout", parents=[common],
        help="revoke and delete the stored API key",
    )

    # run
    run_p = sub.add_parser("run", parents=[common],
                           help="program the FPGA and run @app.local_entrypoint()")
    run_p.add_argument("file", help="Python file defining an App")
    run_p.add_argument("--fpga-id", type=int, default=None, dest="fpga_id",
                       help="override the fpga_id set in the file")
    run_p.add_argument("--no-program", action="store_true", dest="no_program",
                       help="skip programming (FPGA already has the design loaded)")
    run_p.add_argument("--sys-clk", type=lambda s: int(float(s)), default=None,
                       dest="sys_clk_freq", metavar="HZ",
                       help="override the SoC compute clock in Hz, e.g. 90e6 "
                            "(default: build server's 50 MHz)")

    # synth <design.py>  — local, no cloud
    synth_p = sub.add_parser(
        "synth", help="local synthesis report (resource util) — no cloud/board"
    )
    synth_p.add_argument("design", help="user Amaranth design.py")

    # pnr <design.py>  — local, no cloud
    pnr_p = sub.add_parser(
        "pnr", help="local full-SoC place-and-route report (Fmax/timing/util)"
    )
    pnr_p.add_argument("design", help="user Amaranth design.py")
    pnr_p.add_argument("--target-mhz", type=float, default=None, dest="target_mhz",
                       help="timing target; also re-clocks the SoC")

    # status [fpga_id]
    st_p = sub.add_parser("status", parents=[common],
                          help="show FPGA states (all, or one if fpga_id given)")
    st_p.add_argument("fpga_id", nargs="?", type=int, default=None,
                      help="FPGA id (omit for the full table)")
    st_p.add_argument("--json", action="store_true",
                      help="print the raw orchestrator response as JSON")

    # job <fpga_id> <job_id>
    job_p = sub.add_parser("job", parents=[common], help="show job status and metadata")
    job_p.add_argument("fpga_id", type=int)
    job_p.add_argument("job_id")
    job_p.add_argument("--json", action="store_true",
                       help="print the raw orchestrator response as JSON")

    # logs <fpga_id> <job_id>
    log_p = sub.add_parser("logs", parents=[common], help="print build logs for a job")
    log_p.add_argument("fpga_id", type=int)
    log_p.add_argument("job_id")

    # cancel <fpga_id> <job_id>
    can_p = sub.add_parser("cancel", parents=[common], help="cancel a queued job")
    can_p.add_argument("fpga_id", type=int)
    can_p.add_argument("job_id")

    # reset <fpga_id>
    res_p = sub.add_parser("reset", parents=[common],
                           help="reset an FPGA back to idle (reflashes base SoC)")
    res_p.add_argument("fpga_id", type=int)

    # read <fpga_id> <addr>
    read_p = sub.add_parser("read", parents=[common],
                            help="read 32-bit word(s) from a live FPGA")
    read_p.add_argument("fpga_id", type=int)
    read_p.add_argument("address", type=lambda s: int(s, 0),
                        help="byte address (hex or decimal)")
    read_p.add_argument("--count", type=int, default=1)

    # write <fpga_id> <addr> <value>
    wr_p = sub.add_parser("write", parents=[common],
                          help="write a 32-bit word to a live FPGA")
    wr_p.add_argument("fpga_id", type=int)
    wr_p.add_argument("address", type=lambda s: int(s, 0),
                      help="byte address (hex or decimal)")
    wr_p.add_argument("value", type=lambda s: int(s, 0),
                      help="32-bit value (hex or decimal)")

    args = parser.parse_args()

    dispatch = {
        "login":  cmd_login,
        "logout": cmd_logout,
        "run":    cmd_run,
        "status": cmd_status,
        "job":    cmd_job,
        "logs":   cmd_logs,
        "cancel": cmd_cancel,
        "reset":  cmd_reset,
        "read":   cmd_read,
        "write":  cmd_write,
        "synth":  cmd_synth,
        "pnr":    cmd_pnr,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
