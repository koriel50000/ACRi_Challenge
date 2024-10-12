#include <algorithm>

#include "kernel.hpp"

void kernel(
  uint8_t data[SIZE],
  uint64_t code[256],
  uint64_t code_size[256],
  uint8_t out[SIZE*8]
) {
#pragma HLS interface axis port=data
#pragma HLS interface axis port=out
#pragma HLS interface bram port=code
#pragma HLS interface bram port=code_size
#pragma HLS array_partition variable=code
#pragma HLS array_partition variable=code_size

	// 256文字の最大符号長は高々 9 bit？
	// ⇒誤り。出力してみると40 bitを超えることがある。
	uint64_t buf = 0;
	int len = 0;
	int iptr = 0;
	int optr = 0;
	while (iptr < SIZE) {
#pragma HLS pipeline II=1
		// @thanks https://acri-vhls-challenge.web.app/user/nabesan_go/code/vq7toeQWUJ3YvaVNUOqv
		// FIXME ループはパイプライン処理を阻害する？
		if (len < 8) {
			uint8_t ch = data[iptr++];
			uint64_t cd = code[ch];
			uint64_t sz = code_size[ch];
			buf |= cd << (64 - len - sz);
			len += sz;
		}
		if (len >= 8) {
			out[optr++] = buf >> 56;
			buf <<= 8;
			len -= 8;
		}
	}
	while (len > 0) {
#pragma HLS pipeline II=1
		out[optr++] = buf >> 56;
		buf <<= 8;
		len -= 8;
	}
}
