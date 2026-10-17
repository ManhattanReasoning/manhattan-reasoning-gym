<p align="center">
  <img src="https://raw.githubusercontent.com/ManhattanReasoning/manhattan-reasoning-sdk/main/assets/repo_logo.png" alt="Manhattan Reasoning Developer Kit" width="600">
</p>

# Manhattan Reasoning Gym

Python SDK and CLI (`mrg`) for hardware design: iterate **locally** with fast
synthesis / place-and-route feedback, run untrusted agents in a **sandbox**, and
program **real cloud FPGAs** — all from one package.

| Surface | What it does | Needs |
| --- | --- | --- |
| `mrg.build` | local synth / place-and-route reports (no board) | Docker |
| `mrg.Sandbox` | run an agent in a locked container that promotes to silicon | Docker |
| `mrg.cloud` | program + drive a real ECP5 over the cloud | API key |

> Cloud FPGA access requires an **allowlisted API key** (`mrg login`). Local
> synth/PnR and sandboxing need only **Docker** — no toolchain to install.

## Install

```bash
pip install manhattan-reasoning-gym
```

For local synth/PnR and sandboxing, pull the pinned toolchain image once (the
SDK runs it for you — you never install yosys/nextpnr/LiteX):

```bash
docker pull ghcr.io/manhattanreasoning/mrg-sandbox:latest
```

If your system Python is broken or externally managed, the robust path is
`uv venv && uv pip install manhattan-reasoning-gym`.

## Agent quickstart (headless)

Copy-paste path for coding agents and CI — no browser, no prompts:

```bash
export MRG_API_KEY=...        # or: mrg login --github-token <no-scope PAT>
mrg synth examples/design.py  # JSON report on stdout, ~seconds
mrg pnr   examples/design.py  # JSON report (fmax_mhz, timing_met), ~tens of seconds
mrg run   examples/app.py     # programs a real FPGA, ~2-3 min first time
mrg reset 0                   # free the board when done (async, ~1 min)
```

- **All bus addresses are byte addresses**: word *n* in your design's register
  map is byte address `4*n` for `app.read`/`app.write` and `mrg read`/`mrg write`.
- **After `mrg run` the board stays reserved to you** — reuse it with
  `mrg run --no-program` / `mrg read` / `mrg write`, and release it with
  `mrg reset <fpga_id>` when done. Reset is an async queued job that reflashes
  the base SoC (~1 min); poll `mrg status --json` rather than expecting
  instant `idle`.
- `mrg synth` / `mrg pnr` print machine-readable JSON on stdout and exit
  non-zero when the build fails; `mrg status --json`, `mrg jobs --json`, and
  `mrg job --json` emit the raw orchestrator response — parse JSON, don't
  scrape tables. A build in flight has no board yet (see below), so
  `mrg jobs` is the only way to find its `job_id` without already having it.
- Rough time budget: `synth` seconds · `pnr` tens of seconds · `mrg run`
  ~2-3 min for the first program · `--no-program` seconds · `reset` ~1 min.
- `mrg login` with no flags is an interactive GitHub device flow. Headless,
  prefer `$MRG_API_KEY`, or `mrg login --github-token <PAT>` (also reads
  `$GITHUB_TOKEN`) with a **no-scope** PAT — it's only used to read your
  username. If you must run the device flow from a non-tty context, relay the
  printed URL + code to a human.

## Local feedback — `mrg.build`

```python
import manhattan_reasoning_gym as mrg

rep = mrg.build.synth("examples/design.py")   # fast: resource utilization
rep = mrg.build.pnr("examples/design.py")     # full SoC: real Fmax + timing
print(rep.fmax_mhz, rep.timing_met, rep.util.dsp.used)
```

CLI: `mrg synth design.py` · `mrg pnr design.py [--sys-clk-mhz N] [--timing-target-mhz N]`
(`--sys-clk-mhz` re-clocks the SoC; `--timing-target-mhz` is the constraint PnR
optimizes against and `timing_met` is graded on, defaulting to the sys clock;
`--target-mhz` stays as a legacy alias for both).

`synth` is the cheap "is it getting bigger?" signal (core-only util); `pnr`
places & routes the full SoC for the truthful system-clock Fmax and timing.

## Sandboxed agents — `mrg.Sandbox`

Run untrusted agent code in a locked container (`--network none`, no key) that
iterates locally and promotes vetted candidates to silicon:

```python
import manhattan_reasoning_gym as mrg

result = mrg.Sandbox(files=["examples/design.py", "examples/agent.py"]).run("agent.py")
for promo in result.promotions:
    print(promo)
```

The container never holds a key or touches the network; the promote is a file
handoff the trusted host brokers to silicon. **No promote gating by default** —
the agent gates itself; pass `guard=` for an opt-in host-side check. With no API
key, silicon resolves to a no-op backend; set one (and `silicon="cloud"`) for
real hardware.

## Real silicon — `mrg.cloud`

```python
import manhattan_reasoning_gym as mrg

app = mrg.cloud.App("my_design", design="examples/design.py")
with app:                    # programs the FPGA on first use, releases on exit
    app.write(0x0, 3)        # byte address: word n in the design => 4*n
    print("read:", app.read(0x8))
```

`read`/`write` calls go over one WebSocket that the SDK opens on the first
call, reuses for every call after it, and closes after 25 s without a call
(the next call reopens it). On a network that blocks WebSockets, use HTTP
instead: `App(..., transport="http")`, `MRG_TRANSPORT=http`, or
`--transport http` on `mrg run` / `mrg read` / `mrg write`. HTTP works
anywhere but is much slower: each call is a queued job (about 0.5 s or more).

Or drive it from the CLI with a file that defines an `App` and an
`@app.local_entrypoint()` — see [`examples/app.py`](https://github.com/ManhattanReasoning/manhattan-reasoning-sdk/blob/main/examples/app.py), which
exercises the example MAC design end-to-end:

```bash
mrg run examples/app.py
```

Name your registers by subclassing `mrg.RegisterMap` (byte addresses).

**Board lifecycle:** the `with app:` form releases the board on exit, but
`mrg run` does **not** — the reservation persists so you can keep iterating
with `mrg run --no-program`, `mrg read`, and `mrg write`. Free the board with
`mrg reset <fpga_id>` (an async job that reflashes the base SoC, ~1 min — poll
`mrg status`).

## Authentication (cloud only)

API requests authenticate with an opaque API key sent as `X-API-Key`. You obtain
one by exchanging a GitHub identity — the orchestrator verifies you against its
allowlist and mints a key.

```bash
mrg login       # GitHub device flow (or paste a no-scope PAT), then you're set
```

The key is saved to `~/.config/mrg/credentials.json` (mode `0600`, keyed by
orchestrator URL). Every command resolves the key in this order:

1. `--api-key` flag
2. `$MRG_API_KEY`
3. the stored login (`mrg login`)

`mrg logout` revokes the key on the server and clears it locally.

## Tutorials

Two runnable notebooks in [`examples/notebooks/`](https://github.com/ManhattanReasoning/manhattan-reasoning-sdk/tree/main/examples/notebooks):

- **`01_build_and_run`** — local synth/pnr feedback → program a real FPGA.
- **`02_sandboxing`** — run an agent in a locked sandbox that promotes to silicon.

The shared `examples/design.py` is a tiny tunable multiply-accumulate; tweak its
`WIDTH` and watch DSP/Fmax move. `examples/agent.py` is the sandboxed agent, and
`examples/app.py` is the `mrg run` app that drives the MAC on real silicon.

## CLI reference

```
mrg login | logout
mrg synth <design.py>                       # local synthesis report
mrg pnr   <design.py> [--sys-clk-mhz N] [--timing-target-mhz N] [--target-mhz N]
                                            # local full-SoC place & route
mrg run   <file.py> [--no-program [--fpga-id N]]   # --fpga-id only meaningful with --no-program:
                                                     # a fresh build is never pinned to a board
mrg status [fpga_id] [--json]
mrg jobs [--status queued|running|complete|failed|cancelled]
         [--type build|run|reset|all] [-n N | --all] [--json]
                                            # default: your 10 most recent builds
mrg job <job_id> [--json]
mrg logs | cancel <job_id>
mrg reset <fpga_id>
mrg read  <fpga_id> <addr> [--count N]
mrg write <fpga_id> <addr> <value>
```

Human-readable `mrg status` output includes the server commit from `/health`.
A package built with local changes is marked `modified source`. Older servers
or builds without commit metadata show `unknown`; an unavailable health endpoint
shows `unavailable` without suppressing board status. `mrg status --json` keeps
its existing board-only output and does not make the extra health request.

## Development

```bash
pip install -e ".[dev]"
ruff check . && pytest -q
```

## License

MIT — see [LICENSE](https://github.com/ManhattanReasoning/manhattan-reasoning-sdk/blob/main/LICENSE).

### Multi-file cloud designs

Provide the top-level design file. The SDK automatically finds local Python
imports, Verilog module instances, literal `include` headers, and literal
`$readmemh`/`$readmemb` data files under the design's folder:

```python
app = mrg.App("my_design", design="project/design.v")
```

Discovery follows dependencies recursively and uploads only the files it finds.
It reads source text without running your code. Python dependencies already
installed on the server, such as Amaranth, are not uploaded or installed again.

Alternatively, provide a directory containing the supporting files:

```python
app = mrg.App(
    "my_design",
    design="project/design.v",
    include_dir="support",
)
```

`include_dir` is relative to the design file's folder (or can be absolute).
**All supported source files in that directory and its subdirectories are
uploaded**, together with the main design. Use a directory containing only the
files needed for the build. Directory contents become the bundle root; a main
file outside that directory is placed at the root, while a main file inside it
keeps its relative path. For example, use `include_dir=".."` when a design in
`project/rtl/top.v` needs files elsewhere under `project/`.

The directory option also works with Python/Amaranth designs and from the CLI:
`mrg run client.py --include-dir support`. Paths are relative to the design,
not the client script. Hidden entries, symlink directories, `__pycache__`,
`node_modules`, `build`, and `dist` are not scanned.

For precise control, `sources=["rtl/child.v", "include/constants.vh"]` replaces
automatic discovery and directory collection with that exact file list. Paths
are relative to `include_dir` when supplied, otherwise the design's folder.
Repeatable `mrg run client.py --source rtl/child.v` uses this explicit mode too.
`sources=[]` opts out of discovery. Do not list the main file again.

Automatic discovery supports ordinary static imports and Verilog instances;
it is not a full language preprocessor. Dynamic Python imports and generated
file names need directory or explicit-file mode. Macro-based Verilog includes
request one of those modes. Duplicate module definitions or ambiguous include
names require an explicit selection or qualified include path. Missing Verilog
modules are still checked by the compiler. Textually included `.v` files with
literal include names are kept out of the separate compilation list.

Supported files: `.py`, `.v`, `.vh`, `.mem`, `.hex`. Use simple path names
(letters, digits, `_`, `-`, `.`, `/`); source symlinks and dependencies escaping
the selected root are rejected. Bundles are limited to 128 files, 8 MiB
compressed and 16 MiB expanded including metadata. Discovery scans at most
4096 directory entries and 16 MiB of source text; choose a smaller directory
or an explicit list if necessary. A design without local dependencies keeps
the single-file upload format. This requires the matching cloud API/build-image
update; local `mrg synth`/`mrg pnr` remain single-file commands.
See [the example](examples/multi_file/README.md).

### Status durations

`mrg status` shows `HELD FOR` (time since the current reservation began) and
`JOB AGE` (time since submission of the job currently assigned to the board).
When logged in, it also lists your queued/running builds, including those with
no board yet, with elapsed time and the latest stage when available. That extra
lookup has a three-second timeout; a failure does not hide the board table.
Anonymous status requests do not fetch private jobs.

Durations are snapshots in `HH:MM:SS`; run the command again to refresh them.
Job age includes queueing, building, and flashing—it is not billable build time.
`mrg jobs` and `mrg job` also show elapsed time; completed, failed, or cancelled
jobs use their last-update timestamp as the end instead of continuing to tick.
Older servers/sessions without start timestamps show `-`, as do invalid dates.
Client clock skew can affect active durations; negative values are clamped to zero.

`mrg jobs` lists your 10 most recent builds by default, with a footer counting
what's hidden. Every HTTP `app.read()`/`app.write()` is its own `run` job, so
`--type all` is mostly those. `-n N` shows more and `--all` shows everything.
`--json` is always the raw, unfiltered server response.
JSON output retains the API response shape without adding computed duration fields.
