#include "kernel.hpp"
#include <hls_stream.h>
#include <hls_vector.h>
#include <math.h>

const int TOP5 = 5;
const int DEGREE = 64;

typedef float top5_t[TOP5];

template <int M, int N>
void shift(top5_t v, const float f) {
#pragma HLS inline
	for (int i = M; i > N; --i) {
#pragma HLS unroll
		v[i] = v[i - 1];
	}
	v[N] = f;
}

void lshift(top5_t v) {
#pragma HLS inline
	for (int i = 0; i < TOP5 - 1; i++) {
#pragma HLS unroll
		v[i] = v[i + 1];
	}
}

float insert_sort0(top5_t v, const float f) {
	if (true) {
		v[0] = f;
	}
	return v[0];
}

float insert_sort1(top5_t v, const float f) {
	bool b0 = (f > v[0]);
	if (b0) {
		shift<1, 0>(v, f);
	} else {
		v[1] = f;
	}
	return v[1];
}

float insert_sort2(top5_t v, const float f) {
	bool b0 = (f > v[0]);
	bool b1 = (f > v[1]);
	if (b0) {
		shift<2, 0>(v, f);
	} else if (b1) {
		shift<2, 1>(v, f);
	} else {
		v[2] = f;
	}
	return v[2];
}

float insert_sort3(top5_t v, const float f) {
	bool b0 = (f > v[0]);
	bool b1 = (f > v[1]);
	bool b2 = (f > v[2]);
	if (b0) {
		shift<3, 0>(v, f);
	} else if (b1) {
		shift<3, 1>(v, f);
	} else if (b2) {
		shift<3, 2>(v, f);
	} else {
		v[3] = f;
	}
	return v[3];
}

float insert_sort4(top5_t v, const float f) {
	bool b0 = (f > v[0]);
	bool b1 = (f > v[1]);
	bool b2 = (f > v[2]);
	bool b3 = (f > v[3]);
	if (b0) {
		shift<4, 0>(v, f);
	} else if (b1) {
		shift<4, 1>(v, f);
	} else if (b2) {
		shift<4, 2>(v, f);
	} else if (b3) {
		shift<4, 3>(v, f);
	} else {
		v[4] = f;
	}
	return v[4];
}

int quick_max(const top5_t buf[DEGREE]) {
	int p[DEGREE / 2];
	int len = 2;
	for (int i = 0; i < DEGREE; i += len) {
#pragma HLS unroll
		int hi = i;
		int lo = i + 1;
		p[i / 2] = (buf[hi][0] >= buf[lo][0]) ? hi : lo;
	}
	for (int j = 0; j < ilogb(DEGREE) - 1; j++) {
		for (int i = 0; i < DEGREE / 2; i += len) {
#pragma HLS unroll
			int hi = p[i];
			int lo = p[i + len / 2];
			p[i] = (buf[hi][0] >= buf[lo][0]) ? hi : lo;
		}
		len *= 2;
	}

	return p[0];
}

void kernel(const float in[SIZE], float out[TOP5]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE

	top5_t buf[DEGREE];
	float min[DEGREE];
#pragma HLS array_partition variable=buf
#pragma HLS array_partition variable=min

	int ptr = 0;
	for (int d = 0; d < DEGREE; d++) {
#pragma HLS unroll
		min[d] = insert_sort0(buf[d], in[ptr++]);
	}
	for (int d = 0; d < DEGREE; d++) {
#pragma HLS unroll
		min[d] = insert_sort1(buf[d], in[ptr++]);
	}
	for (int d = 0; d < DEGREE; d++) {
#pragma HLS unroll
		min[d] = insert_sort2(buf[d], in[ptr++]);
	}
	for (int d = 0; d < DEGREE; d++) {
#pragma HLS unroll
		min[d] = insert_sort3(buf[d], in[ptr++]);
	}
	for (int d = 0; d < DEGREE; d++) {
#pragma HLS unroll
		min[d] = insert_sort4(buf[d], in[ptr++]);
	}

	for (int i = TOP5; i < SIZE / DEGREE; i++) {
#pragma HLS pipeline
		for (int d = 0; d < DEGREE; d++) {
#pragma HLS unroll
			float f = in[ptr++];
			if (f > min[d]) {
				min[d] = insert_sort4(buf[d], f);
			}
		}
	}

	for (int i = 0; i < TOP5; i++) {
#pragma HLS pipeline
		int max_d = quick_max(buf);
		out[i] = buf[max_d][0];
		lshift(buf[max_d]);
	}
}
