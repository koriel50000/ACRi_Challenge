#include "kernel.hpp"

typedef struct {
	bool end;
	float data;
} fifo_t;

void read_input(hls::stream<float>& stream_data, hls::stream<bool>& stream_end, hls::stream<fifo_t>& outs) {
	fifo_t fifo;
	while (!stream_end.read()) {
		fifo.end = false;
		fifo.data = stream_data.read();
		outs << fifo;
	}
	fifo.end = true;
	outs << fifo;
}

void write_result(float* output, hls::stream<fifo_t>& outs) {
	float acc = 0;
	while (true) {
		fifo_t fifo;
		outs >> fifo;
		if (fifo.end) break;
		acc += fifo.data;
	}
	*output = acc;
}

// @see Vitis 高位合成ユーザー ガイド
// https://japan.xilinx.com/support/documentation/sw_manuals_j/xilinx2020_1/ug1399-vitis-hls.pdf
// Vitis HLS ライブラリ リファレンス > HLS ストリーム ライブラリ
void kernel(hls::stream<float>& stream_data, hls::stream<bool>& stream_end, float* output) {
	hls::stream<fifo_t> outs("output_fifo");

#pragma HLS dataflow
	read_input(stream_data, stream_end, outs);
	write_result(output, outs);
}
