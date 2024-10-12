#include "kernel.hpp"

void kernel(const float in[SIZE * SIZE], float out[SIZE * SIZE]) {
// @thanks https://acri-vhls-challenge.web.app/user/Fogrex/code/9fuEjpYs83OqLsc2b1uR
// @see Vivado HLS 最適化手法ガイド (UG1270)
// https://docs.xilinx.com/v/u/ja-JP/ug1270-vivado-hls-opt-methodology-guide
// FIXME pragma HLS interface axis/bram
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in block factor=SIZE
#pragma HLS array_partition variable=out cyclic factor=SIZE

	for (int j = 0; j < SIZE; j++) {
#pragma HLS pipeline II=1
		for (int i = 0; i < SIZE; i++) {
//#pragma HLS unroll
			out[j * SIZE + i] = in[i * SIZE + j];
		}
	}
}
