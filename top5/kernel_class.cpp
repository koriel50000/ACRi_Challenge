#include "kernel.hpp"
#include <hls_stream.h>
#include <hls_vector.h>

const int TOP5 = 5;
const int DEGREE = 8;

class sorted_t {
private:
	hls::vector<float, TOP5> v;
	int ptr;
	template <int N> void shift(const float f);
public:
	sorted_t();
	float get() { return v[ptr]; }
	float next() { return v[ptr++]; }
	void insert_sort(const float f);
};

sorted_t::sorted_t() {
	for (int i = 0; i < TOP5; i++) {
#pragma HLS unroll
		v[i] = -2000.0;
	}
	ptr = 0;
}

template <int N>
void sorted_t::shift(const float f) {
#pragma HLS inline
	for (int i = 4; i > N; --i) {
#pragma HLS unroll
		v[i] = v[i - 1];
	}
	v[N] = f;
}

void sorted_t::insert_sort(const float f) {
	if (f > v[0]) {
		shift<0>(f);
	} else if (f > v[1]) {
		shift<1>(f);
	} else if (f > v[2]) {
		shift<2>(f);
	} else if (f > v[3]) {
		shift<3>(f);
	} else if (f > v[4]) {
		shift<4>(f);
	}
}

// @see Vitis 高位合成ユーザー ガイド
// https://japan.xilinx.com/support/documentation/sw_manuals_j/xilinx2020_1/ug1399-vitis-hls.pdf
float quick_max(sorted_t buf[DEGREE]) {
	int tmp4[4];
	for (int i = 0, d = 0; i < 4; i++) {
#pragma HLS unroll
		int hi = d++;
		int lo = d++;
		tmp4[i] = (buf[hi].get() >= buf[lo].get()) ? hi : lo;
	}
	int tmp2[2];
	for (int i = 0, d = 0; i < 2; i++) {
#pragma HLS unroll
		int hi = tmp4[d++];
		int lo = tmp4[d++];
		tmp2[i] = (buf[hi].get() >= buf[lo].get()) ? hi : lo;
	}
	int hi = tmp2[0];
	int lo = tmp2[1];
	int max_degree = (buf[hi].get() >= buf[lo].get()) ? hi : lo;

	return buf[max_degree].next();
}

void kernel(const float in[SIZE], float out[TOP5]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE

	// @see 配列の初期化
	sorted_t buf[DEGREE];

	for (int i = 0; i < SIZE; i += DEGREE) {
#pragma HLS pipeline
		for (int d = 0; d < DEGREE; d++) {
#pragma HLS unroll
			buf[d].insert_sort(in[i + d]);
		}
	}

	for (int i = 0; i < TOP5; i++) {
#pragma HLS pipeline
		out[i] = quick_max(buf);
	}
}
