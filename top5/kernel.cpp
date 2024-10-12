#include "kernel.hpp"
#include <hls_vector.h>

const int TOP5 = 5;
const int DEGREE = 16;

typedef hls::vector<float, TOP5> top5_t;

// @thanks https://acri-vhls-challenge.web.app/user/NapoliN/code/drGwk8PYbQtvTOGsaFbN
void insert_sort(top5_t& v, const float f) {
	float t0 = v[0];
	float t1 = v[1];
	float t2 = v[2];
	float t3 = v[3];
	float t4 = v[4];
	v[0] = (f <= t0) ? t0 : f;
	v[1] = (f <= t1) ? t1 : ((f < t0) ? f : t0);
	v[2] = (f <= t2) ? t2 : ((f < t1) ? f : t1);
	v[3] = (f <= t3) ? t3 : ((f < t2) ? f : t2);
	v[4] = (f <= t4) ? t4 : ((f < t3) ? f : t3);
}

float lshift(top5_t& v) {
	float t0 = v[0];
	v[0] = v[1];
	v[1] = v[2];
	v[2] = v[3];
	v[3] = v[4];
	return t0;
}

void kernel(const float in[SIZE], float out[5]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE

	static top5_t buf[DEGREE];
#pragma HLS array_partition variable=buf

	for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
		buf[j] = -2000.0;
	}

	for (int i = 0; i < SIZE; i += DEGREE) {
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			insert_sort(buf[j], in[i + j]);
		}
	}

	static top5_t top = -2000.0;

	for (int i = 0; i < TOP5; i++) {
#pragma HLS pipeline
		for (int j = 0; j < DEGREE; j++) {
			insert_sort(top, lshift(buf[j]));
		}
		out[i] = lshift(top);
	}
}
