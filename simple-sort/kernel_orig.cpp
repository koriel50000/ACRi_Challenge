#include "kernel.hpp"

#define UP(a,b) if (a > b) { int t = a; a = b; b = t; }
#define DN(a,b) if (a < b) { int t = a; a = b; b = t; }

const int CHUNK_SIZE = 8;

using buffer_t = int[SIZE];
using chunk_t = int[CHUNK_SIZE];

void sort0202(chunk_t b) {
	UP(b[0], b[2]);
	UP(b[1], b[3]);
	DN(b[4], b[6]);
	DN(b[5], b[7]);
}

void sort0201(chunk_t b) {
	UP(b[0], b[1]);
	UP(b[2], b[3]);
	DN(b[4], b[5]);
	DN(b[6], b[7]);
}

void sort0303(chunk_t b) {
	UP(b[0], b[4]);
	UP(b[1], b[5]);
	UP(b[2], b[6]);
	UP(b[3], b[7]);
}

void sort0302(chunk_t b) {
	UP(b[0], b[2]);
	UP(b[1], b[3]);
	UP(b[4], b[6]);
	UP(b[5], b[7]);
}

void sort0301(chunk_t b) {
	UP(b[0], b[1]);
	UP(b[2], b[3]);
	UP(b[4], b[5]);
	UP(b[6], b[7]);
}

// @see https://ja.wikipedia.org/wiki/バイトニックソート
// @thanks https://edom18.hateblo.jp/entry/2020/09/21/150416
// 「バイトニックソートの実装を理解する」
void bitonic_merger8(chunk_t buf) {
	sort0202(buf);
	sort0201(buf);
	sort0303(buf);
	sort0302(buf);
	sort0301(buf);
}

template <int N>
struct merger_t {
	static void merge_sort(buffer_t& ibuf, buffer_t& obuf, const int p) {
		int i = p;
		int j = p + N / 2;
		int k = p;
		while (i < p + N / 2 && j < p + N) {
			if (ibuf[i] < ibuf[j]) {
				obuf[k++] = ibuf[i++];
			} else {
				obuf[k++] = ibuf[j++];
			}
		}
		if (i == p + N / 2) {
			while (j < p + N) {
				obuf[k++] = ibuf[j++];
			}
		} else {
			while (i < p + N / 2) {
				obuf[k++] = ibuf[i++];
			}
		}
	}

	static void compute_sort(buffer_t& ibuf, buffer_t& obuf) {
		for (int p = 0; p < SIZE; p += N) {
			merger_t<N>::merge_sort(ibuf, obuf, p);
		}
	}
};

void bitonic_sort(buffer_t& buf) {
	for (int k = 0; k < SIZE; k += CHUNK_SIZE) {
		bitonic_merger8(&buf[k]);
	}
}

void bitonic_input(const int in[SIZE], buffer_t& buf) {
	for (int i = 0; i < SIZE; i += 4) {
		int i0 = in[i + 0];
		int i1 = in[i + 1];
		buf[i + 0] = (i0 > i1) ? i1 : i0;
		buf[i + 1] = (i0 < i1) ? i0 : i1;
		int i2 = in[i + 2];
		int i3 = in[i + 3];
		buf[i + 2] = (i2 > i3) ? i3 : i2;
		buf[i + 3] = (i2 < i3) ? i2 : i3;
	}
}

void write_result(buffer_t& buf, int out[SIZE]) {
	int i = 0;
	int j = SIZE / 2;
	int k = 0;
	while (i < SIZE / 2 && j < SIZE) {
		if (buf[i] < buf[j]) {
			out[k++] = buf[i++];
		} else {
			out[k++] = buf[j++];
		}
	}
	if (i == SIZE / 2) {
		while (j < SIZE) {
			out[k++] = buf[j++];
		}
	} else {
		while (i < SIZE / 2) {
			out[k++] = buf[i++];
		}
	}
}

// FIXME 回路の複雑さとサイクル数のバランスで実行時間が決まる？
// ログを読めれば最適値がわかるようになるのか？
void kernel(const int in[SIZE], int out[SIZE]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=4
#pragma HLS array_partition variable=out cyclic factor=4

	buffer_t buf;
#pragma HLS array_partition variable=buf cyclic factor=4
	buffer_t ibuf, obuf;

	bitonic_input(in, buf);
	bitonic_sort(buf);
	merger_t<16>::compute_sort(buf, ibuf);
	merger_t<32>::compute_sort(ibuf, obuf);
	merger_t<64>::compute_sort(obuf, ibuf);
	merger_t<128>::compute_sort(ibuf, obuf);
	merger_t<256>::compute_sort(obuf, ibuf);
	merger_t<512>::compute_sort(ibuf, obuf);
	write_result(obuf, out);
}
