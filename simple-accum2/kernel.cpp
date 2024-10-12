#include "kernel.hpp"
#include <ap_fixed.h>

const int SIZE = 1024;
const int DEGREE = 128;
const int NUM = 8;

// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/HJ8rcDdY1Y7lizCR7Zez
// @see ug1399 Arbitrary Precision Fixed-Point Data Types
using fixed16_t = ap_fixed<32, 16>;
using chunk_t = fixed16_t[DEGREE];

// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/HJ8rcDdY1Y7lizCR7Zez
// @see ug1399 Recursive Functions
// In C++, templates can implement tail recursion and can then be used for synthesizable tail-recursive designs.
// N = 2, 4, 8, 16, ...
template <typename T, int N>
struct recursive_sum {
	static T compute(const T v[]) {
		T t[N / 2];
#pragma HLS array_partition variable=t
		for (int i = 0; i < N; i += 2) {
#pragma HLS unroll
			t[i / 2] = v[i] + v[i + 1];
		}
		return recursive_sum<T, N / 2>::compute(t);
	}
};

template <typename T>
struct recursive_sum<T, 1> {
	static T compute(const T v[]) {
		return v[0];
	}
};

template <int N>
using quick_sum = recursive_sum<fixed16_t, N>;

void kernel(const float in[SIZE], const int size, float *out) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=size
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE

	// @see ug1399 HLS Programmers Guide
	//  Initializing and Resetting Arrays
	static fixed16_t acc[NUM];
#pragma HLS array_partition variable=acc

	for (int i = 0; i < SIZE; i += DEGREE) {
#pragma HLS pipeline
		chunk_t buf;
#pragma HLS array_partition variable=buf
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			buf[j] = (i + j < size) ? in[i + j] : 0;
		}
		acc[i / DEGREE] = quick_sum<DEGREE>::compute(buf);
	}

	*out = quick_sum<NUM>::compute(acc);
}
