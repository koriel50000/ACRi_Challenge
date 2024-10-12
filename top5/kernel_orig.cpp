#include "kernel.hpp"

const int TOP5 = 5;
const int DEGREE = 8;

typedef float top5_t[TOP5];

template <int N>
void shift(float* v, const float f) {
        for (int i = 4; i > N; --i) {
                v[i] = v[i - 1];
        }
        v[N] = f;
}

// FIXME 関数を列挙せずに固定値で回路を作成するには？
void shift0(top5_t top5, const float f) {
	// FIXME 一時変数を用いないswap,rotateの方法は？
	top5[4] = top5[3];
	top5[3] = top5[2];
	top5[2] = top5[1];
	top5[1] = top5[0];
	top5[0] = f;
}

void shift1(top5_t top5, const float f) {
	top5[4] = top5[3];
	top5[3] = top5[2];
	top5[2] = top5[1];
	top5[1] = f;
}

void shift2(top5_t top5, const float f) {
	top5[4] = top5[3];
	top5[3] = top5[2];
	top5[2] = f;
}

void shift3(top5_t top5, const float f) {
	top5[4] = top5[3];
	top5[3] = f;
}

void shift4(top5_t top5, const float f) {
	top5[4] = f;
}

void insert_sort(top5_t top5, const float f) {
	bool b0 = (f > top5[0]);
	bool b1 = (f > top5[1]);
	bool b2 = (f > top5[2]);
	bool b3 = (f > top5[3]);
	bool b4 = (f > top5[4]);
	if (b0) {
		shift0(top5, f);
	} else if (b1) {
		shift1(top5, f);
	} else if (b2) {
		shift2(top5, f);
	} else if (b3) {
		shift3(top5, f);
	} else if (b4) {
		shift4(top5, f);
	}
}

int quick_max(const float buf[DEGREE * 5], const int ptr[DEGREE]) {
	int tmp4[4];
	for (int i = 0, d = 0; i < 4; i++) {
		int hi = d++;
		int lo = d++;
		tmp4[i] = (buf[ptr[hi]] >= buf[ptr[lo]]) ? hi : lo;
	}
	int tmp2[2];
	for (int i = 0, d = 0; i < 2; i++) {
		int hi = tmp4[d++];
		int lo = tmp4[d++];
		tmp2[i] = (buf[ptr[hi]] >= buf[ptr[lo]]) ? hi : lo;
	}
	int hi = tmp2[0];
	int lo = tmp2[1];
	int max_degree = (buf[ptr[hi]] >= buf[ptr[lo]]) ? hi : lo;

	return max_degree;
}

void kernel(const float in[SIZE], float out[5]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE
#pragma HLS array_partition variable=out

	static float buf[DEGREE * 5];
	static int ptr[DEGREE];
#pragma HLS array_partition variable=buf

	for (int d = 0; d < DEGREE; d++) {
#pragma HLS unroll
		ptr[d] = d * 5;
		for (int i = 0; i < 5; i++) {
			buf[d * 5 + i] = -2000.0;
		}
	}

	top5_loop: for (int i = 0; i < SIZE; i += DEGREE) {
		float chunk[DEGREE];
		load_loop: for (int d = 0; d < DEGREE; d++) {
			chunk[d] = in[i + d];
		}
		degree_loop: for (int d = 0; d < DEGREE; d++) {
			insert_sort(&buf[d * 5], chunk[d]);
		}
	}

	max_loop: for (int i = 0; i < 5; i++) {
		int max_degree = quick_max(buf, ptr);
		out[i] = buf[ptr[max_degree]++];
	}
}
