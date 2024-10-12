#include "kernel.hpp"

void kernel(const float in[SIZE * SIZE], float out[SIZE * SIZE]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=SIZE
#pragma HLS array_partition variable=out block factor=SIZE
// @thanks https://acri-vhls-challenge.web.app/user/offNaria/code/EgM0u7me8WUBxbwNqSmC

	// @thanks https://acri-vhls-challenge.web.app/user/NapoliN/code/q2XsGkgp9ZYHWwIPKYpz
	int iptr = 0;
	for (int j = 0; j < SIZE; j++) {
#pragma HLS pipeline rewind
		int optr = j;
		for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
			out[optr] = in[iptr++];
			optr += 256;
		}
	}
}
