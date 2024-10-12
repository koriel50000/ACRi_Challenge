#include "kernel.hpp"
#include <ap_fixed.h>

const int SIZE = 1024;
const int DEGREE = 128;
const int NUM = 8;

// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/raDvqvlBejKsSfhiWMuR
// @see ug1399 Arbitrary Precision Fixed-Point Data Types
using mfixed_t = ap_fixed<32, 18>;
using vfixed_t = ap_fixed<32, 25>;

using chunk_t = float[DEGREE];
template <int N>
using mchunk_t = mfixed_t[N];
template <int N>
using vchunk_t = vfixed_t[N];

// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/raDvqvlBejKsSfhiWMuR
// @see ug1399 Recursive Functions
// In C++, templates can implement tail recursion and can then be used for synthesizable tail-recursive designs.
// N = 2, 4, 8, 16, ...
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

	static T sum(const chunk_t v) {
		T t[DEGREE];
#pragma HLS array_partition variable=t
		for (int i = 0; i < DEGREE; i++) {
#pragma HLS unroll
			t[i] = v[i];
		}
		return recursive_sum<T, DEGREE>::compute(t);
	}

	static T expsum(const chunk_t v) {
		T t[DEGREE];
#pragma HLS array_partition variable=t
		for (int i = 0; i < DEGREE; i++) {
#pragma HLS unroll
			t[i] = v[i] * v[i];
		}
		return recursive_sum<T, DEGREE>::compute(t);
	}
};

template <typename T>
struct recursive_sum<T, 1> {
	static T compute(const T v[]) {
		return v[0];
	}
};

template <int N>
using mean_sum = recursive_sum<mfixed_t, N>;
template <int N>
using vari_sum = recursive_sum<vfixed_t, N>;

void kernel(
  const float in[SIZE],
  const int size,
  float mean[1],
  float vari[1]
) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=size
#pragma HLS interface axis port=mean
#pragma HLS interface axis port=vari
#pragma HLS array_partition variable=in cyclic factor=DEGREE

#pragma HLS pipeline

	static mchunk_t<NUM> sums;
	static vchunk_t<NUM> exps;
#pragma HLS array_partition variable=sums
#pragma HLS array_partition variable=exps

	for (int i = 0; i < SIZE; i += DEGREE) {
		chunk_t buf;
#pragma HLS array_partition variable=buf
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
	// @thanks https://acri-vhls-challenge.web.app/user/fpga/code/M8sNwWiPptiEnUn8GdOu
			buf[j] = (i + j < size) ? in[i + j] : 0;
		}
		sums[i / DEGREE] = mean_sum<DEGREE>::sum(buf);
		exps[i / DEGREE] = vari_sum<DEGREE>::expsum(buf);
	}

	float sum = mean_sum<NUM>::compute(sums);
	float exp = vari_sum<NUM>::compute(exps);

	// @thanks https://acri-vhls-challenge.web.app/user/nabesan_go/code/rHfUa5w2mfKIapFojgv6
	mean[0] = sum / size;
	vari[0] = (exp - sum * sum / size) / size;
}
