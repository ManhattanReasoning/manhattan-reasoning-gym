# Multi-file Verilog example

`design.v` instantiates the module in `rtl/constant_value.v`, which includes
`include/values.vh`. The SDK sends these three explicitly selected files.

After the corresponding server and firmware updates are deployed:

```sh
mrg run examples/multi_file/client.py
```

The client reads back `0x12345678`. This command builds and programs a real FPGA.
Supporting file paths are relative to the folder containing `design.v`, even
when the client is launched from another working directory.

For Python/Amaranth projects, use the same `sources=[...]` option to include
helper modules and package `__init__.py` files. Imports run on the build server;
only the listed source files are uploaded. No dependencies are installed.
