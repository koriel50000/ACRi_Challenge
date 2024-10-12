#include "kernel.hpp"

template <int N>
struct merger_t {
	static void merge_sort(int buf[SIZE], const int p) {
		int tmp[N];
		int i = p;
		int j = p + N / 2;
		int k = 0;
		while (i < p + N / 2 && j < p + N) {
			if (buf[i] < buf[j]) {
				tmp[k++] = buf[i++];
			} else {
				tmp[k++] = buf[j++];
			}
		}
		if (i == p + N / 2) {
			while (j < p + N) {
				tmp[k++] = buf[j++];
			}
		} else {
			while (i < p + N / 2) {
				tmp[k++] = buf[i++];
			}
		}
		for (int l = 0; l < N; l++) {
#pragma HLS unroll
			buf[p + l] = tmp[l];
		}
	}

	static void compute_sort(int buf[SIZE]) {
		for (int p = 0; p < SIZE; p += N) {
#pragma HLS dependence variable=buf inter false
			merger_t<N>::merge_sort(buf, p);
		}
	}
};

void read_input(const int in[SIZE], int buf[SIZE]) {
	for (int i = 0; i < SIZE; i += 2) {
#pragma HLS unroll factor=4 skip_exit_check
		int lo = in[i + 0];
		int hi = in[i + 1];
		buf[i + 0] =  (lo < hi) ? lo : hi;
		buf[i + 1] =  (lo < hi) ? hi : lo;
	}
}

void write_result(int buf[SIZE], int out[SIZE]) {
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

void kernel(const int in[SIZE], int out[SIZE]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=8

	int buf[SIZE];

	read_input(in, buf);
	merger_t<4>::compute_sort(buf);
	merger_t<8>::compute_sort(buf);
	merger_t<16>::compute_sort(buf);
	merger_t<32>::compute_sort(buf);
	merger_t<64>::compute_sort(buf);
	merger_t<128>::compute_sort(buf);
	merger_t<256>::compute_sort(buf);
	merger_t<512>::compute_sort(buf);
	write_result(buf, out);
}
