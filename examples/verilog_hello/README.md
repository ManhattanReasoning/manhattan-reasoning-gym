# verilog_hello

Same design as [`hello_wishbone`](../hello_wishbone/) — a 512 × 32-bit
echo memory — but written as plain Verilog instead of Amaranth, to
demonstrate submitting a hand-written `.v` design directly.

## Contents

- `design.v` — `echo_slave`: the memory behind the Wishbone B4 slave
  contract (registered single-cycle ack), identical timing to
  `hello_wishbone/design.py`'s `EchoSlave`
- `client_sdk.py` — `mrg run` entrypoint: writes a 4-word pattern over
  the SDK, reads it back, verifies byte-for-byte
- `client.py` — end-to-end smoke test over the raw bridge protocol
  (identical to `hello_wishbone/client.py`; the protocol doesn't care
  which language flashed the board)
- `tests/` — local synth report smoke test (no simulator; see below)

## Run

```sh
# Local synth report only (no cloud, no hardware)
mrg synth examples/verilog_hello/design.v

# Against a live node
mrg run examples/verilog_hello/client_sdk.py
# -> OK: pattern written and read back intact

# Or, once flashed, the raw protocol directly:
python client.py --host 192.168.1.101
```

## The Wishbone contract, for a Verilog top module

Unlike an Amaranth design (where the top-level class is found by scanning
for the one `Elaboratable` exposing the right port *attributes*), a plain
Verilog file's top module is found by scanning for the one module whose
port list matches this contract by name, width, and direction:

| Port | Width | Direction |
| --- | --- | --- |
| `clk` | 1 | input |
| `rst` | 1 | input |
| `wb_cyc` | 1 | input |
| `wb_stb` | 1 | input |
| `wb_we` | 1 | input |
| `wb_adr` | 9 | input |
| `wb_dat_w` | 32 | input |
| `wb_sel` | 4 | input |
| `wb_dat_r` | 32 | output |
| `wb_ack` | 1 | output |

If a file has more than one module matching this contract, pass
`--top <name>` (CLI) or `top="<name>"` (`App(...)`/`mrg.build.synth`/`pnr`)
to disambiguate — `design.v` here only has one, so it's auto-detected.

## Address map

Nothing to document: byte offsets 0x000–0x7FC are 512 plain words of
storage. Reads return whatever was last written.

## Note: no simulation tests here

`hello_wishbone/tests/` uses Amaranth's own Python simulator
(`amaranth.sim.Simulator`), which only simulates Amaranth's IR — it can't
run a hand-written `.v` file. There's currently no Verilog simulator wired
into `mrg_build` (Icarus Verilog and Verilator ship in the sandbox image's
toolchain bundle already, but nothing calls them yet), so `tests/` here is
a synth-report smoke test instead of a real simulation.
