"""Run a sandboxed agent from the command line.

    # mock silicon (no key, no boards): iterate an agent fully offline
    python -m manhattan_reasoning_gym.bench run \
        --file design.py --file agent.py --entry agent.py

    # real boards: the key stays in THIS trusted process, never in the container
    MRG_API_KEY=... python -m manhattan_reasoning_gym.bench run \
        --file design.py --file agent.py --entry agent.py --silicon cloud

This is the thin CLI over :class:`manhattan_reasoning_gym.bench.Sandbox`; the
promote broker is wired up for you.
"""

from __future__ import annotations

import argparse
import sys

from .sandbox import Sandbox


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mrg-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run an agent in a sandbox")
    run.add_argument("--file", action="append", default=[], dest="files",
                     metavar="PATH", help="file to copy into the sandbox (repeatable)")
    run.add_argument("--entry", required=True, help="entrypoint to run (e.g. agent.py)")
    run.add_argument("--silicon", default="auto",
                     choices=["auto", "cloud", "mock"],
                     help="silicon backend (default: auto)")
    run.add_argument("--dev", action="store_true",
                     help="trusted 'dev' isolation (network on, key forwarded)")
    run.add_argument("--api-url", default=None)
    run.add_argument("--sys-clk", type=lambda s: int(float(s)), default=None,
                     help="compute-clock override in Hz for silicon builds")
    run.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args(argv)

    sb = Sandbox(
        files=args.files,
        isolation="dev" if args.dev else "locked",
        silicon=args.silicon,
        api_url=args.api_url,
        sys_clk_freq=args.sys_clk,
    )
    print(f"[bench] running {args.entry} (silicon={args.silicon})", file=sys.stderr)
    result = sb.run(args.entry, timeout=args.timeout)

    print(result.stdout.rstrip())
    if not result.ok:
        print(result.stderr[-2000:], file=sys.stderr)
    for promo in result.promotions:
        outcome = promo.get("silicon", promo.get("reason"))
        print(f"[bench] promote {promo.get('id')}: "
              f"accepted={promo.get('accepted')} {outcome}", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
