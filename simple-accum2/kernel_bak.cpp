#include "kernel.hpp"

const int DEGREE = 128;
const int SIZE = 1024;
const int NUM = 8;

using chunk_t = float[DEGREE];

float quick_sum(chunk_t v) {
#pragma HLS pipeline
	const int p = 7; //ilogb(DEGREE);
	chunk_t t;
#pragma HLS array_partition variable=t

	for (int i = 0; i < DEGREE; i++) {
#pragma HLS unroll
		t[i] = v[i];
	}
	for (int j = 0, d = 1; j < p; j++, d *= 2) {
		for (int i = 0; i + d < DEGREE; i += d * 2) {
#pragma HLS unroll
			t[i] += t[i + d];
		}
	}
	return t[0];
}

void kernel(const float in[SIZE], const int size, float *out) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=size
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE

	// @see ug1399 HLS Programmers Guide
	//  Initializing and Resetting Arrays
	static float acc[NUM];
#pragma HLS array_partition variable=acc

	for (int i = 0; i < SIZE; i += DEGREE) {
#pragma HLS pipeline
		chunk_t buf;
#pragma HLS array_partition variable=buf
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			buf[j] = (i + j < size) ? in[i + j] : 0;
		}
		acc[i / DEGREE] = quick_sum(buf);
	}

	*out = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
		+ ((acc[4] + acc[5]) + (acc[6] + acc[7]));
}
