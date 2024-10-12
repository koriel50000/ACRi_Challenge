#include "kernel.hpp"
#include <hls_math.h>

using buffer_t = int[SIZE];

void bitonic_merger(int* bi, int* bo, int p, int q, int d) {
	printf("void sort%02d%02d(buffer_t b) {\n",
			p + 1, q + 1);
	for (int i = 0; i < SIZE; i++) {
#pragma HLS pipeline
		if ((i & d) == 0) {
			bool up = ((i >> p) & 2) == 0;
			int j = i | d;
			printf("\t%s(b[%d], b[%d]);\n",
					(up ? "UP" : "DN"), i, j, i, j);
			if ((bi[i] > bi[i | d]) == up) {
				bo[i] = bi[i | d];
				bo[i | d] = bi[i];
			} else {
				bo[i] = bi[i];
			}
		} else {
			bo[i] = bi[i];
		}
	}
	printf("}\n\n");
}

void bitonic_sort(int* buf0, int* buf1) {
	int *bi, *bo, *tmp;
	bi = buf0;
	bo = buf1;
	for (int p = 0; p < ilogb(SIZE); p++) {
		for (int q = 0; q <= p; q++) {
			int d = 1 << (p - q);
			bitonic_merger(bi, bo, p, q, d);
			tmp = bi;
			bi = bo;
			bo = tmp;
		}
	}
	printf("void bitonic_sort(buffer_t buf) {\n");
	bool flag = true;
	for (int p = 0; p < ilogb(SIZE); p++) {
		for (int q = 0; q <= p; q++) {
			printf("\tsort%02d%02d(%s);\n",
					p + 1, q + 1,
					flag ? "buf" : "buf",
					flag ? "buf1" : "buf0");
			flag = !flag;
		}
	}
	printf("}\n\n");
	fflush(stdout);
}

void kernel(const int in[SIZE], int out[SIZE]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=SIZE
#pragma HLS array_partition variable=out cyclic factor=SIZE

	buffer_t buf0, buf1;
#pragma HLS array_partition variable=buf0
#pragma HLS array_partition variable=buf1

	for (int i = 0; i < SIZE; i++) {
		buf0[i] = in[i];
	}

	bitonic_sort(buf0, buf1);

	for (int i = 0; i < SIZE; i++) {
		out[i] = buf0[i];
	}
}
