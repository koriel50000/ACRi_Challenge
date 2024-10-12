#include "kernel.hpp"

void kernel(const float in[1024], float out[1024], int size) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=out cyclic factor=16

	loop: for (int i = 0; i < size; i++) {
#pragma HLS UNROLL factor=16 skip_exit_check
#pragma HLS PIPELINE
		out[i] = in[i] * 2;
        }
}
