#include "kernel.hpp"
#include <hls_math.h>

const int HALF = 40;
const int CHUNK = SIZE + HALF;

// @see HD, Figure 3-3
constexpr int clp2(int x) {
	x = x - 1;
	x = x | (x >> 1);
	x = x | (x >> 2);
	x = x | (x >> 4);
	x = x | (x >> 8);
	x = x | (x >> 16);
	return x + 1;
}

template <typename T, int M>
T muladd(T *mat, T vec[M]) {
	const int N = clp2(M);
	const int p = ilogb(N);
	T t[M];
#pragma HLS array_partition variable=t

	for (int i = 0; i < M; i++) {
#pragma HLS unroll
		t[i] = mat[i] * vec[i];
	}
	for (int j = 0, d = 1; j < p; j++, d *= 2) {
		for (int i = 0; i + d < M; i += d * 2) {
#pragma HLS unroll
			t[i] += t[i + d];
		}
	}
	return t[0];
}

unsigned int offset(unsigned int i) {
	static const unsigned int table[] {
		0, 40, 80, 0, 40, 80, 0, 40, 80,
		0, 40, 80, 0, 40, 80, 0, 40, 80,
		0, 40, 80, 0, 40, 80, 0, 40, 80,
		0, 40, 80, 0, 40, 80, 0, 40, 80,
		0, 40, 80, 0, 40, 80, 0, 40, 80,
		0, 40, 80, 0, 40, 80, 0, 40, 80,
		0, 40, 80, 0, 40, 80, 0, 40, 80,
		0, 40, 80, 0, 40, 80, 0, 40, 80,
		0, 40, 80, 0, 40, 80, 0, 40,
	};
	return table[i];
}

void lshift(float mat[SIZE * 2]) {
#pragma HLS inline
	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		mat[i] = mat[i + SIZE];
	}
}

void kernel(
  const float in_mat[SIZE * SIZE],
  const float in_vec[SIZE],
  float out[SIZE]
) {
#pragma HLS interface axis port=in_mat
#pragma HLS interface axis port=in_vec
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in_mat cyclic factor=CHUNK
#pragma HLS array_partition variable=in_vec
#pragma HLS array_partition variable=out

	float vec[SIZE];
#pragma HLS array_partition variable=vec

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		vec[i] = in_vec[i];
	}

	float mat[SIZE * 2];
#pragma HLS array_partition variable=mat

	int ptr = 0;
	for (int i = 0; i < SIZE; i++) {
#pragma HLS pipeline off
		lshift(mat);
		int o = offset(i);
		if (o <= HALF) {
			for (int j = 0; j < CHUNK; j++) {
#pragma HLS unroll
				mat[o + j] = in_mat[ptr++];
			}
		}

		out[i] = muladd<float, SIZE>(mat, vec);
	}
}
