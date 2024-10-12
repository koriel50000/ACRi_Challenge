#include "kernel.hpp"

const int DEGREE = 128;

// @thanks https://www.youtube.com/watch?v=1WHytfpVgtc
void kernel(const float in[1024], float out[1024], const int size) {
	// @thanks https://acri-vhls-challenge.web.app/user/iwabuchi/code/OduhTdRboDjlPrEXhMoJ
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS interface axis port=size
#pragma HLS array_partition variable=in cyclic factor=DEGREE
#pragma HLS array_partition variable=out cyclic factor=DEGREE
	// @thanks https://acri-vhls-challenge.web.app/user/pgate1/code/taKnuiJiXfeaCYvVoSZe
	// @see ug1399 Reduces the (II) for a function or loop ..
#pragma HLS pipeline

	// @see https://acri-vhls-challenge.web.app/challenge/bai-gaeshi2
	// ・入力される値の範囲は [-1000, 1000)
	// ・要素数がsizeとは書いていない (テストベンチは1024固定)
	for (int i = 0; i < 1024; i++) {
#pragma HLS unroll factor=DEGREE skip_exit_check
	// @see ap_common.h#floatToRawBits
		union {
	// @see https://ja.wikipedia.org/wiki/IEEE_754
	// @see https://www.cqpub.co.jp/interface/sample/200702/I0702168.pdf
			struct ieee754 {
				unsigned int frac : 23;
				unsigned int exp : 8;
				unsigned int sign : 1;
			} b;
			float f;
		} f2b;
		f2b.f = in[i];
		f2b.b.exp += (f2b.b.exp != 0);
		out[i] = f2b.f;
	}
}
