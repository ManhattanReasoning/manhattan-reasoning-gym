#!/usr/bin/env python3
"""The agent — runs INSIDE the locked-down sandbox (no network, no key).

It iterates locally with fast synth/pnr feedback, and only when a candidate
passes its own local gate does it promote it to silicon. The promote is a file
handoff to the trusted host (the Sandbox, outside the container); the agent never
touches the cloud or holds a key.
"""

import manhattan_reasoning_gym as mrg

DESIGN = "/work/design.py"


def show(tag: str, r) -> None:
    u = r.util
    line = f"[agent] {tag}: ok={r.ok}"
    if r.fmax_mhz is not None:
        line += f" fmax={r.fmax_mhz:.1f} timing_met={r.timing_met}"
    line += f" dsp={u.dsp.used} ff={u.ff.used} bram={u.bram.used}"
    print(line, flush=True)


def main() -> None:
    print("[agent] iterating locally (offline, no credentials)...", flush=True)

    # Tier 1 — synth: cheap util check.
    show("synth", mrg.build.synth(DESIGN))

    # Tier 2 — pnr: the truthful system Fmax + timing. This is the agent's gate.
    report = mrg.build.pnr(DESIGN)
    show("pnr", report)

    if report.fits and report.timing_met:
        print("[agent] local gate PASSED -> promoting to silicon", flush=True)
        resp = mrg.sandbox.promote(DESIGN, report, agent="demo-agent")
        if resp["accepted"]:
            print(f"[agent] SILICON accepted -> {resp['silicon']}", flush=True)
        else:
            print(f"[agent] promote REJECTED: {resp['reason']}", flush=True)
    else:
        print("[agent] local gate failed; not spending a board", flush=True)


if __name__ == "__main__":
    main()
