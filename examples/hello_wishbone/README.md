# hello_wishbone

Minimal Wishbone slave that echoes writes back as reads — the
recommended starting point for verifying the interface contract.

A single 512 × 32-bit block RAM filling the whole user design region:
write any word, read it back. No registers, no logic, no surprises.
If this design works on a node, everything below it (network path,
bridge firmware, bus wiring) is proven; debug your own design's logic,
not the plumbing.

Ported from the ecp5-ethernet-soc prototype, where it was the
hardware-verified echo workload used for the original latency
measurements.

## Contents

- `design.py` — `EchoSlave`: the memory behind the Wishbone B4 slave
  contract (registered single-cycle ack)
- `client.py` — end-to-end smoke test: writes a 512-word pattern over
  the bridge protocol, reads it back, verifies byte-for-byte
- `tests/` — simulation tests of the interface contract (pytest +
  Amaranth simulator, no hardware)

## Run

```sh
# Simulation (no hardware)
pytest tests/

# Against a live node
python client.py --host 192.168.1.101
# -> OK: 512 words written and read back intact
```

## Address map

Nothing to document: byte offsets 0x000–0x7FC are 512 plain words of
storage. Reads return whatever was last written.
