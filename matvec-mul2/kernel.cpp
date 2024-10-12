#include "kernel.hpp"
#include <ap_fixed.h>

// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/Mt9Dxlhh2Syr3DpzusI0
const int PARALLEL = 5;

// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/Mt9Dxlhh2Syr3DpzusI0
// @see ug1399 Arbitrary Precision Fixed-Point Data Types
using fixed11_t = ap_fixed<32, 11>;
using chunk_t = fixed11_t[SIZE];

// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/Mt9Dxlhh2Syr3DpzusI0
// @see ug1399 Recursive Functions
// In C++, templates can implement tail recursion and can then be used for synthesizable tail-recursive designs.
// N = 5, 10, 15, 20, ...
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
struct recursive_sum<T, 5> {
	static T compute(const T v[]) {
		return ((v[0] + v[1]) + (v[2] + v[3])) + v[4];
	}
};

template <int N>
using quick_sum = recursive_sum<fixed11_t, N>;

void kernel(
  const float in_mat[SIZE * SIZE],
  const float in_vec[SIZE],
  float out[SIZE]
) {
#pragma HLS interface axis port=in_mat
#pragma HLS interface axis port=in_vec
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in_mat cyclic factor=SIZE * PARALLEL
#pragma HLS array_partition variable=in_vec
#pragma HLS array_partition variable=out cyclic factor=PARALLEL

	// @see ug1399 HLS Programmers Guide
	//  Initializing and Resetting Arrays
	static chunk_t vec;
#pragma HLS array_partition variable=vec

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		vec[i] = in_vec[i];
	}

	for (int i = 0, j = 0; j < SIZE; i += SIZE, j++) {
#pragma HLS pipeline
#pragma HLS unroll factor=PARALLEL skip_exit_check
		chunk_t buf;
#pragma HLS array_partition variable=buf
		for (int k = 0; k < SIZE; k++) {
#pragma HLS unroll
			fixed11_t mat = in_mat[i + k];
			buf[k] = mat * vec[k];
		}
		out[j] = quick_sum<SIZE>::compute(buf);
	}
}
