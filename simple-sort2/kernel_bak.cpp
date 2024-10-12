#include "kernel.hpp"
#include <stdio.h>

// FIXME swapのよい記述方法は？
#define SWAP(a,b) if (a > b) { int t = a; a = b; b = t; }

const int N = 7; // N = ilogb(SIZE);

using buffer_t = int[SIZE];

void bitonic_merger(buffer_t b, const int p, const int q) {
printf("void sort%02d%02d(buffer_t b) {\n", p + 1, q + 1);
	for (int i = 0; i < SIZE; i++) {
// FIXME なぜ手動で展開した結果と同じにならないのか？
		if ((i & (1 << q)) == 0) {
			const int j = i | (1 << q);
			if ((i & (2 << p)) == 0) {
				SWAP(b[i], b[j]);
printf("\tSWAP(b[%d], b[%d]);\n", i, j);
			} else {
				SWAP(b[j], b[i]);
printf("\tSWAP(b[%d], b[%d]);\n", j, i);
			}
		}
        }
printf("}\n\n");
}

// @see https://ja.wikipedia.org/wiki/バイトニックソート
// @thanks https://edom18.hateblo.jp/entry/2020/09/21/150416
// 「バイトニックソートの実装を理解する」
void bitonic_sort(buffer_t buf) {
	for (int p = 0; p < N; p++) {
#pragma HLS unroll
		for (int q = p; q >= 0; --q) {
			bitonic_merger(buf, p, q);
		}
	}
}

void kernel(const int in[SIZE], int out[SIZE]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in
#pragma HLS array_partition variable=out

	buffer_t buf;
#pragma HLS array_partition variable=buf

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		buf[i] = in[i];
	}

	bitonic_sort(buf);

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		out[i] = buf[i];
	}
}
