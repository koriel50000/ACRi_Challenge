#include "kernel.hpp"
#include <ap_int.h>

const int AFFINE_SIZE = 256;
const int CLASSES = 10;

void kernel(int in[256], int weight[10 * 256], int out[10]) {
#pragma HLS interface bram port=in
#pragma HLS interface axis port=weight
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=32
#pragma HLS array_partition variable=weight cyclic factor=AFFINE_SIZE

	int ptr = 0;
	for (int j = 0; j < CLASSES; j++) {
#pragma HLS pipeline
		ap_int<11> acc = 0;
		for (int i = 0; i < AFFINE_SIZE; i++) {
			acc += in[i] * weight[ptr++];
		}
		out[j] = acc;
	}
}
