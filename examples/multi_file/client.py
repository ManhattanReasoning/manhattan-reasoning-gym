"""Run from the SDK repository root with `mrg run examples/multi_file/client.py`."""

from pathlib import Path

import manhattan_reasoning_gym as mrg

app = mrg.App(
    "multi_file",
    design=str(Path(__file__).with_name("design.v")),
)


@app.local_entrypoint()
def main():
    with app:
        assert app.read(0) == 0x12345678
        print("Multi-file design returned the expected value")
