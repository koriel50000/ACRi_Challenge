#pragma once
#include "types.hpp"

int16_t mul(const uint4_t v, const uint4_t w);

template <int C>
int16_t muladd(const int c, const int_t<C> vu, const int_t<C> wi) {
	static int16_t t[C];
#pragma HLS array_partition variable=t

	for (int i = 0; i < C; i++) {
#pragma HLS unroll
		if (i >= c) break;
		t[i] = mul(vu[i], wi[i]);
	}

	for (int d = 1; d < C; d *= 2) {
		if (d >= c) break;
		for (int i = 0; i < C; i += d * 2) {
#pragma HLS unroll
			if (i >= c) break;
			t[i] += t[i + d];
		}
	}
	return t[0];
}

uint4_t batch_norm_relu(const int16_t acc, const int16_t thr[]);

uint4_t batch_norm(const int16_t acc, const int16_t thr[]);
