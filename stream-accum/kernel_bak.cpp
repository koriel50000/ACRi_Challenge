#include "kernel.hpp"
#include <math.h>

void read_input(hls::stream<float>& stream_data, hls::stream<bool>& stream_end, hls::stream<float>& outs) {
	while (!stream_end.read()) {
#pragma HLS pipeline II=1
		outs << stream_data.read();
	}
	outs << NAN;
}

void write_result(float* output, hls::stream<float>& outs) {
	float acc = 0;
	while (true) {
#pragma HLS pipeline II=1
		float data;
		outs >> data;
		if (isnan(data)) break;
		acc += data;
	}
	*output = acc;
}

// @see Vitis 高位合成ユーザー ガイド
// https://japan.xilinx.com/support/documentation/sw_manuals_j/xilinx2020_1/ug1399-vitis-hls.pdf
// Vitis HLS ライブラリ リファレンス > HLS ストリーム ライブラリ
void kernel(hls::stream<float>& stream_data, hls::stream<bool>& stream_end, float* output) {
	hls::stream<float> outs("output_fifo");

#pragma HLS dataflow
	read_input(stream_data, stream_end, outs);
	write_result(output, outs);
}
