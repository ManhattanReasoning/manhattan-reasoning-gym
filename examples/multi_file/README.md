# Multi-file Verilog example

`design.v` instantiates the module in `rtl/constant_value.v`, which includes
`include/values.vh`. The SDK discovers these dependencies from the top-level
file; the client does not list them separately.

After the corresponding server and firmware updates are deployed:

```sh
mrg run examples/multi_file/client.py
```

The client reads back `0x12345678`. This command builds and programs a real FPGA.
Discovery starts in the folder containing `design.v`, even when the client is
launched from another working directory.

For Python/Amaranth projects, local imports are discovered recursively too,
including package `__init__.py` files. Discovery does not execute those files.
No additional third-party dependencies are installed.

Use `include_dir="support"` on `App` (or `--include-dir support` on `mrg run`)
to send all supported source files in a directory instead. That path is relative
to the design's folder. `sources=[...]` remains an explicit override for projects
that need an exact list. See the main README for discovery limits and layouts.
