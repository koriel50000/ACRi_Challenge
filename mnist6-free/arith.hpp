#pragma once
#include "types.hpp"

int16_t mul(const uint4_t v, const uint4_t w) {
	static const int16_t v0[] = {
		0, 1, 2, 3, 4, 6, 8, 12,
		0, -1, -2, -3, -4, -6, -8, -12,
	};
#pragma HLS array_partition variable=v

	ap_uint<1> sign = v[3] ^ w[3];
	int16_t oval = v0[(sign, v(2, 0))] * (w(2, 0) > 0);
	return oval << (w(2, 0) - 1);
}

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

uint4_t batch_norm_relu(const int16_t acc, const int16_t thr[]) {
	ap_uint<1> b0 = acc < thr[0];
	ap_uint<1> b1 = acc < thr[1];
	ap_uint<1> b2 = acc < thr[2];
	ap_uint<1> b3 = acc < thr[3];
	ap_uint<1> b4 = acc < thr[4];
	ap_uint<1> b5 = acc < thr[5];
	ap_uint<1> b6 = acc < thr[6];
	ap_uint<8> bits = (1, b6, b5, b4, b3, b2, b1, b0);
	// @see UG1399, Vitis HLS Coding Styles > Functions > C/C++ Builtin Functions
	return __builtin_ctz(bits);
}

uint4_t batch_norm(const int16_t acc, const int16_t thr[]) {
	static const uint4_t indexTable[] = {
		7, 6, 5, 2, 4, 10, 1, 12, 0, 3, 9, 11, 15, 0, 14, 13
	};
#pragma HLS array_partition variable=indexTable

	ap_uint<1> b0 = acc < thr[0];
	ap_uint<1> b1 = acc < thr[1];
	ap_uint<1> b2 = acc < thr[2];
	ap_uint<1> b3 = acc < thr[3];
	ap_uint<1> b4 = acc < thr[4];
	ap_uint<1> b5 = acc < thr[5];
	ap_uint<1> b6 = acc < thr[6];
	ap_uint<1> b7 = acc < thr[7];
	ap_uint<1> b8 = acc < thr[8];
	ap_uint<1> b9 = acc < thr[9];
	ap_uint<1> b10 = acc < thr[10];
	ap_uint<1> b11 = acc < thr[11];
	ap_uint<1> b12 = acc < thr[12];
	ap_uint<1> b13 = acc < thr[13];
	ap_uint<15> bits = (0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13);
	// @see HD, Figure 5-26. Number of trailing zeros using a de Brujin cycle.
	// https://en.wikipedia.org/wiki/De_Bruijn_sequence
	return indexTable[((bits + 1) * 0x09af)(15, 12)];
}
