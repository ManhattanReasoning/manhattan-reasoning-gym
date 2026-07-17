// Minimal Wishbone B4 slave: a 512 x 32-bit memory, hand-written Verilog.
//
// Stores words on write; returns the same words on read. A plain-Verilog
// port of examples/hello_wishbone/design.py's EchoSlave, with identical
// timing, so the two examples are directly comparable:
//
//   Cycle N   : master asserts cyc+stb. For a write, the write fires
//               combinatorially so the data is captured at the rising
//               edge of cycle N+1.
//   Cycle N+1 : ack goes high. For a read, dat_r is valid here because
//               the registered read latches mem[adr_N] at this edge.
//   Cycle N+2 : ack clears.
//
// sel is accepted but ignored; all accesses are full 32-bit words.
//
// The port list is the fixed Wishbone contract every Verilog user design
// must expose (see cloud_fpga_firmware.export.VERILOG_WB_PORTS): exactly
// these 10 ports, these widths, these directions.
module echo_slave (
    input  wire        clk,
    input  wire        rst,
    input  wire        wb_cyc,
    input  wire        wb_stb,
    input  wire        wb_we,
    input  wire [8:0]  wb_adr,   // 512 words = 9 address bits
    input  wire [31:0] wb_dat_w,
    input  wire [3:0]  wb_sel,   // accepted, ignored
    output wire [31:0] wb_dat_r,
    output reg         wb_ack
);
    reg [31:0] mem [0:511];
    reg [31:0] dat_r_reg;
    wire wr_en = wb_cyc & wb_stb & wb_we & ~wb_ack;

    always @(posedge clk) begin
        if (rst) begin
            wb_ack <= 1'b0;
            dat_r_reg <= 32'b0;
        end else begin
            wb_ack <= wb_cyc & wb_stb & ~wb_ack;
            if (wr_en)
                mem[wb_adr] <= wb_dat_w;
            dat_r_reg <= mem[wb_adr];
        end
    end

    assign wb_dat_r = dat_r_reg;
endmodule
