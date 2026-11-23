# CNN_FPGA_Shapes (Vivado 2026.1, xc7z020clg400-1)

Open `CNN_FPGA_Shapes.xpr`. The HLS IP is found through the relative path
`../CNN_Shapes/CNN_Shapes/hls/impl/ip` (kept in this repo), so keep the folder layout.
Generated folders (.runs, .cache, .gen) are not included: re-run synthesis/implementation.

- PL clock: 83.333 MHz (HLS closed timing at 12 ns; routed WNS +2.393 ns)
- Image buffer: axi_bram_ctrl_0 BRAM at 0x40000000 (8K, 60x60 int16 Q8.8 = 7200 bytes)
- ds_cnn control registers: 0x40010000 (input_img 0x10/0x14, pred_out 0x1C, logits 0x30)
- Prebuilt hardware export: ../../hardware/CNN_shapes_wrapper.xsa
