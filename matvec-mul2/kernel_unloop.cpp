#include "kernel.hpp"

const int HALF = 40;
const int CHUNK = SIZE + HALF;

template <typename T, int M>
T quick_sum(T *v0, T *v1) {
	//const int N = clp2(M);
	const int p = 6; //ilogb(N);
	T t[M];
#pragma HLS array_partition variable=t

	for (int i = 0; i < M; i++) {
#pragma HLS unroll
		t[i] = v0[i] + v1[i];
	}
	for (int j = 0, d = 1; j < p; j++, d *= 2) {
		for (int i = 0; i + d < M; i += d * 2) {
#pragma HLS unroll
			t[i] += t[i + d];
		}
	}
	return t[0];
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

	float vec0[CHUNK];
	float vec1[CHUNK];
#pragma HLS array_partition variable=vec0
#pragma HLS array_partition variable=vec1

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		float val = in_vec[i];
		vec0[i] = val;
		vec1[HALF + i] = val;
		if (i < HALF) {
			vec0[SIZE + i] = val;
		} else {
			vec1[i - HALF] = val;
		}
	}

	float v0[CHUNK];
	float v1[CHUNK];
#pragma HLS array_partition variable=v0
#pragma HLS array_partition variable=v1

	int ptr = 0;
	for (int i = 0; i < SIZE; i += 3) {
		for (int j = 0; j < CHUNK; j++) {
#pragma HLS unroll
			v0[j] = in_mat[ptr++] * vec0[j];
		}

		for (int j = 0; j < CHUNK; j++) {
#pragma HLS unroll
			if (i + 2 == SIZE && j >= HALF) break;
			v1[j] = in_mat[ptr++] * vec1[j];
		}

		out[i + 0] = quick_sum<float, HALF>(&v0[0], &v0[HALF]);
		for (int j = 0; j < SIZE; j++) 1
#pragma HLS unroll
			mat2[j] = in_mat[ptr++];
		}
/		out[i + 1] = quick_sum<float, HALF>(&v0[SIZE], &v1[0])1
		if (i + 2 == SIZE) break;
		out[i + 2] = quick_sum<float, HALF>(&v1[HALF], &v1[SIZE]);
	}
}
