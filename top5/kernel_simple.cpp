#include "kernel.hpp"
#include <hls_math.h>

const int TOP5 = 5;
const int DEGREE = 8;

typedef float top5_t[TOP5];

void shift0(top5_t v, const float f) {
	v[4] = v[3];
	v[3] = v[2];
	v[2] = v[1];
	v[1] = v[0];
	v[0] = f;
}

void shift1(top5_t v, const float f) {
	v[4] = v[3];
	v[3] = v[2];
	v[2] = v[1];
	v[1] = f;
}

void shift2(top5_t v, const float f) {
	v[4] = v[3];
	v[3] = v[2];
	v[2] = f;
}

void shift3(top5_t v, const float f) {
	v[4] = v[3];
	v[3] = f;
}

void shift4(top5_t v, const float f) {
	v[4] = f;
}

void insert_sort(top5_t v, const float f) {
	bool b0 = (f > v[0]);
	bool b1 = (f > v[1]);
	bool b2 = (f > v[2]);
	bool b3 = (f > v[3]);
	bool b4 = (f > v[4]);
	if (b0) {
		shift0(v, f);
	} else if (b1) {
		shift1(v, f);
	} else if (b2) {
		shift2(v, f);
	} else if (b3) {
		shift3(v, f);
	} else if (b4) {
		shift4(v, f);
	}
}

int quick_max(const float buf[TOP5 * DEGREE], const int ptr[DEGREE]) {
	int p[DEGREE / 2];
	int len = 2;
	for (int i = 0; i < DEGREE; i += len) {
		int hi = i;
		int lo = i + 1;
		p[i / 2] = (buf[ptr[hi]] >= buf[ptr[lo]]) ? hi : lo;
	}

	for (int i = 0; i < DEGREE / 2; i += len) {
		int hi = p[i];
		int lo = p[i + len / 2];
		p[i] = (buf[ptr[hi]] >= buf[ptr[lo]]) ? hi : lo;
	}

	len *= 2;
	for (int i = 0; i < DEGREE / 2; i += len) {
		int hi = p[i];
		int lo = p[i + len / 2];
		p[i] = (buf[ptr[hi]] >= buf[ptr[lo]]) ? hi : lo;
	}

	return p[0];
}

void kernel(const float in[SIZE], float out[5]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE
#pragma HLS array_partition variable=out

	static float buf[TOP5 * DEGREE];
	static int ptr[DEGREE];
#pragma HLS array_partition variable=buf
//#pragma HLS array_partition variable=ptr

	for (int d = 0; d < DEGREE; d++) {
#pragma HLS unroll
		for (int i = 0; i < TOP5; i++) {
			buf[TOP5 * d + i] = -2000.0;
		}
		ptr[d] = TOP5 * d;
	}

	for (int i = 0; i < SIZE; i += DEGREE) {
		float chunk[DEGREE];
		for (int d = 0; d < DEGREE; d++) {
			chunk[d] = in[i + d];
		}
		for (int d = 0; d < DEGREE; d++) {
			insert_sort(&buf[TOP5 * d], chunk[d]);
		}
	}

	for (int i = 0; i < TOP5; i++) {
		int max_d = quick_max(buf, ptr);
		out[i] = buf[ptr[max_d]++];
	}
}
