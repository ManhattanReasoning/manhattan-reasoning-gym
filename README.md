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

## Local feedback — `mrg.build`

```python
import manhattan_reasoning_gym as mrg

rep = mrg.build.synth("examples/design.py")   # fast: resource utilization
rep = mrg.build.pnr("examples/design.py")     # full SoC: real Fmax + timing
print(rep.fmax_mhz, rep.timing_met, rep.util.dsp.used)
```

CLI: `mrg synth design.py` · `mrg pnr design.py [--target-mhz N]`.

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
    app.write(0x0, 3)
    print("read:", app.read(0x8))
```

Or drive it from the CLI: `mrg run my_design.py`.

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

Two runnable notebooks in [`examples/notebooks/`](examples/notebooks/):

- **`01_build_and_run`** — local synth/pnr feedback → program a real FPGA.
- **`02_sandboxing`** — run an agent in a locked sandbox that promotes to silicon.

The shared `examples/design.py` is a tiny tunable multiply-accumulate; tweak its
`WIDTH` and watch DSP/Fmax move. `examples/agent.py` is the sandboxed agent.

## CLI reference

```
mrg login | logout
mrg synth <design.py> [--target-mhz N]      # local synthesis report
mrg pnr   <design.py> [--target-mhz N]      # local full-SoC place & route
mrg run   <file.py> [--fpga-id N] [--no-program]
mrg status [fpga_id]
mrg job | logs | cancel <fpga_id> <job_id>
mrg reset <fpga_id>
mrg read  <fpga_id> <addr> [--count N]
mrg write <fpga_id> <addr> <value>
```

## Development

```bash
pip install -e ".[dev]"
ruff check . && pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
