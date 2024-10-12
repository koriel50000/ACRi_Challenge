#include "kernel.hpp"

// FIXME ログの見方がわからないのでトライ＆エラーになるのがイタい
const int CHUNK_SIZE = 4;

// @see Vitis 高位合成ユーザー ガイド
// https://japan.xilinx.com/support/documentation/sw_manuals_j/xilinx2020_1/ug1399-vitis-hls.pdf
// Vitis HLS ライブラリ リファレンス > HLS ストリーム ライブラリ
void kernel(hls::stream<float>& stream_data, hls::stream<bool>& stream_end, float* output) {
	float acc;

	static float chunk[CHUNK_SIZE];
#pragma HLS array_partition variable=chunk

	for (int i = 0; i < CHUNK_SIZE; i++) {
#pragma HLS unroll
		chunk[i] = 0.0;
	}

	while (true) {
// @thanks https://acri-vhls-challenge.web.app/user/hashi0203/code/o15Zw6wgCnzeo9gnXoQe
#pragma HLS pipeline II=1
		if (stream_end.read()) break;
		chunk[0] += stream_data.read();
		if (stream_end.read()) break;
		chunk[1] += stream_data.read();
		if (stream_end.read()) break;
		chunk[2] += stream_data.read();
		if (stream_end.read()) break;
		chunk[3] += stream_data.read();
		if (stream_end.read()) break;
	}

	acc = (chunk[0] + chunk[1]) + (chunk[2] + chunk[3]);

	*output = acc;
}
