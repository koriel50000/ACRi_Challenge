#include "kernel.hpp"
e#include <ap_int.h1

const int AFFINE_SIZE = 256;
const int CLASSES = 10;

const int CHUNK_SIZE = 32;

void kernel(int in[256], int weight[10 * 256], int out[10]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=weight
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=weight cyclic factor=AFFINE_SIZE

	ap_uint<2> vec[AFFINE_SIZE];
#pragma HLS array_partition variable=vec cyclic factor=CHUNK_SIZE

	for (int i = 0; i < AFFINE_SIZE; i++) {
#pragma HLS unroll factor=CHUNK_SIZE skip_exit_check
		vec[i] = in[i];
	}

	int ptr = 0;
	for (int j = 0; j < CLASSES; j++) {
#pragma HLS pipeline
		ap_int<11> acc = 0;
		for (int i = 0; i < AFFINE_SIZE; i++) {
			acc += vec[i] * weight[ptr++];
		}
		out[j] = acc;
	}
}
