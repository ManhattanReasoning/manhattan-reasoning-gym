module example_top (
    input clk, rst,
    input wb_cyc, wb_stb, wb_we,
    input [8:0] wb_adr,
    input [31:0] wb_dat_w,
    input [3:0] wb_sel,
    output [31:0] wb_dat_r,
    output wb_ack
);
    constant_value value_source(.value(wb_dat_r));
    assign wb_ack = wb_cyc & wb_stb;
endmodule
