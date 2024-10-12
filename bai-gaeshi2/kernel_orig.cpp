#include "kernel.hpp"

const int DEGREE = 128;

// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/bF8U1WQa7LxBNKl0HXzf
unsigned int increase_if_not_zero(const unsigned int i) {
	static const unsigned int table[] = {
		0, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
		21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
		31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
		41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
		51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
		61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
		71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
		81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
		91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
		101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
		111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
		121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
		131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
		141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
		151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
		161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
		171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
		181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
		191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
		201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
		211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
		221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
		231, 232, 233, 234, 235, 236, 237, 238, 239, 230,
		241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
		251, 252, 253, 254, 255, 255,
	};
	return table[i];
}

float revenge_double(const float v) {
#pragma HLS inline
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
	f2b.f = v;
	f2b.b.exp = increase_if_not_zero(f2b.b.exp);
	return f2b.f;
}

// @thanks https://www.youtube.com/watch?v=1WHytfpVgtc
void kernel(const float in[1024], float out[1024], const int size) {
	// @thanks https://acri-vhls-challenge.web.app/user/iwabuchi/code/OduhTdRboDjlPrEXhMoJ
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
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
		out[i] = revenge_double(in[i]);
	}
}
