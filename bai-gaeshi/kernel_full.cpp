#include "kernel.hpp"

void kernel(const float in[1024], float out[1024], int size) {
	// @thanks https://acri-vhls-challenge.web.app/user/iwabuchi/code/OduhTdRboDjlPrEXhMoJ
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=128
#pragma HLS ARRAY_PARTITION variable=out cyclic factor=128

	loop: for (int i = 0; i < size; i++) {
		// @thanks https://www.youtube.com/watch?v=1WHytfpVgtc
#pragma HLS UNROLL factor=128 skip_exit_check
#pragma HLS PIPELINE
		// @thanks https://acri-vhls-challenge.web.app/user/Ichiro/code/OcPJbi5jZ5BjKwUzLXGb
		// @see ap_common.h#floatToRawBits
		union {
			unsigned int b;
			float f;
		} f2b;
		f2b.f = in[i];
		// @thanks https://acri-vhls-challenge.web.app/user/pgate1/code/taKnuiJiXfeaCYvVoSZe
		if (f2b.b != 0) {
			f2b.b += (1 << 23);
		}
		out[i] = f2b.f;
        }
}
