#include "kernel.hpp"

// @see Vitis 高位合成ユーザー ガイド
// https://japan.xilinx.com/support/documentation/sw_manuals_j/xilinx2020_1/ug1399-vitis-hls.pdf
// Vitis HLS ライブラリ リファレンス > HLS ストリーム ライブラリ
void kernel(hls::stream<float>& stream_data, hls::stream<bool>& stream_end, float* output) {
	float acc = 0;
	while (true) {
#pragma HLS pipeline II=1
		if (stream_end.read()) break;
		acc += stream_data.read();
	}
	*output = acc;
}
