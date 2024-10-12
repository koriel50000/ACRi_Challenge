#include "kernel.hpp"

#define UP(a,b,c,d) { c = (a > b) ? b : a; d = (a > b) ? a : b; }
#define DN(a,b,c,d) { c = (a < b) ? b : a; d = (a < b) ? a : b; }

const int CHUNK_SIZE = 8;

using buffer_t = int[SIZE];

void sort0101(buffer_t bi, buffer_t bo) {
	UP(bi[0], bi[1], bo[0], bo[1]);
	DN(bi[2], bi[3], bo[2], bo[3]);
	UP(bi[4], bi[5], bo[4], bo[5]);
	DN(bi[6], bi[7], bo[6], bo[7]);
}

void sort0201(buffer_t bi, buffer_t bo) {
	UP(bi[0], bi[2], bo[0], bo[2]);
	UP(bi[1], bi[3], bo[1], bo[3]);
	DN(bi[4], bi[6], bo[4], bo[6]);
	DN(bi[5], bi[7], bo[5], bo[7]);
}

void sort0202(buffer_t bi, buffer_t bo) {
	UP(bi[0], bi[1], bo[0], bo[1]);
	UP(bi[2], bi[3], bo[2], bo[3]);
	DN(bi[4], bi[5], bo[4], bo[5]);
	DN(bi[6], bi[7], bo[6], bo[7]);
}

void sort0301(buffer_t bi, buffer_t bo) {
	UP(bi[0], bi[4], bo[0], bo[4]);
	UP(bi[1], bi[5], bo[1], bo[5]);
	UP(bi[2], bi[6], bo[2], bo[6]);
	UP(bi[3], bi[7], bo[3], bo[7]);
}

void sort0302(buffer_t bi, buffer_t bo) {
	UP(bi[0], bi[2], bo[0], bo[2]);
	UP(bi[1], bi[3], bo[1], bo[3]);
	UP(bi[4], bi[6], bo[4], bo[6]);
	UP(bi[5], bi[7], bo[5], bo[7]);
}

void sort0303(buffer_t bi, buffer_t bo) {
	UP(bi[0], bi[1], bo[0], bo[1]);
	UP(bi[2], bi[3], bo[2], bo[3]);
	UP(bi[4], bi[5], bo[4], bo[5]);
	UP(bi[6], bi[7], bo[6], bo[7]);
}

void bitonic_sort(buffer_t buf0, buffer_t buf1) {
	sort0101(buf0, buf1);
	sort0201(buf1, buf0);
	sort0202(buf0, buf1);
	sort0301(buf1, buf0);
	sort0302(buf0, buf1);
	sort0303(buf1, buf0);
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
#pragma HLS unroll
		buf0[i] = in[i];
	}

	bitonic_sort(buf0, buf1);

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		out[i] = buf0[i];
	}
}
