#include "kernel.hpp"
#include <hls_vector.h>
#include <hls_stream.h>

const int SIZE = 1024;
const int DEGREE = 4;

typedef hls::vector<int, DEGREE> inbuf_t;
typedef hls::vector<int, SIZE / 2> outbuf_t;
typedef hls::stream<inbuf_t> ififo_t;
typedef hls::stream<outbuf_t> ofifo_t;

typedef hls::stream<hls::vector<int, 8>> pipe8_t;
typedef hls::stream<hls::vector<int, 16>> pipe16_t;
typedef hls::stream<hls::vector<int, 32>> pipe32_t;
typedef hls::stream<hls::vector<int, 64>> pipe64_t;
typedef hls::stream<hls::vector<int, 128>> pipe128_t;
typedef hls::stream<hls::vector<int, 256>> pipe256_t;

void swap(inbuf_t& buf, int p, int q) {
#pragma HLS inline
	int tmp = buf[p];
	buf[p] = buf[q];
	buf[q] = tmp;
}

void sort4(inbuf_t& buf) {
	if (buf[0] > buf[1]) {
		swap(buf, 0, 1);
	}
	if (buf[2] > buf[3]) {
		swap(buf, 2, 3);
	}
	if (buf[0] > buf[2]) {
		swap(buf, 0, 2);
	}
	if (buf[1] > buf[3]) {
		swap(buf, 1, 3);
	}
	if (buf[1] > buf[2]) {
		swap(buf, 1, 2);
	}
}

template <int N>
struct merger_t {
	static void merge_sort(hls::vector<int, N / 2>& lo, hls::vector<int, N / 2>& hi, hls::vector<int, N>& out) {
		int i = 0;
		int j = 0;
		int ptr = 0;
		while (i < N / 2 && j < N / 2) {
			if (lo[i] < hi[j]) {
				out[ptr++] = lo[i++];
			} else {
				out[ptr++] = hi[j++];
			}
		}
		if (i == N / 2) {
			while (j < N / 2) {
				out[ptr++] = hi[j++];
			}
		} else {
			while (i < N / 2) {
				out[ptr++] = lo[i++];
			}
		}
	}

	static void compute_sort(hls::stream<hls::vector<int, N / 2>>& ins, hls::stream<hls::vector<int, N>>& outs) {
		for (int i = 0; i < SIZE / N; i++) {
#pragma HLS pipeline
			hls::vector<int, N / 2> lo = ins.read();
			hls::vector<int, N / 2> hi = ins.read();
			hls::vector<int, N> out;
			merger_t<N>::merge_sort(lo, hi, out);
			outs.write(out);
		}
	}
};

void merge_sort(outbuf_t& lo, outbuf_t& hi, int out[SIZE]) {
	int i = 0;
	int j = 0;
	int ptr = 0;
	while (i < SIZE / 2 && j < SIZE / 2) {
		if (lo[i] < hi[j]) {
			out[ptr++] = lo[i++];
		} else {
			out[ptr++] = hi[j++];
		}
	}
	if (i == SIZE / 2) {
		while (j < SIZE / 2) {
			out[ptr++] = hi[j++];
		}
	} else {
		while (i < SIZE / 2) {
			out[ptr++] = lo[i++];
		}
	}
}

void read_input(const int in[SIZE], ififo_t& ins) {
	for (int i = 0; i < SIZE; i += DEGREE) {
		inbuf_t buf;
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			buf[j] = in[i + j];
		}
		sort4(buf);
		ins.write(buf);
	}
}

void write_result(int out[SIZE], ofifo_t& outs) {
	outbuf_t lo = outs.read();
	outbuf_t hi = outs.read();
	merge_sort(lo, hi, out);
}

void kernel(const int in[SIZE], int out[SIZE]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE

	ififo_t ins("input_fifo");
	pipe8_t ps8("pipe8_fifo");
	pipe16_t ps16("pipe16_fifo");
	pipe32_t ps32("pipe32_fifo");
	pipe64_t ps64("pipe64_fifo");
	pipe128_t ps128("pipe128_fifo");
	pipe256_t ps256("pipe256_fifo");
	ofifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input(in, ins);
	merger_t<8>::compute_sort(ins, ps8);
	merger_t<16>::compute_sort(ps8, ps16);
	merger_t<32>::compute_sort(ps16, ps32);
	merger_t<64>::compute_sort(ps32, ps64);
	merger_t<128>::compute_sort(ps64, ps128);
	merger_t<256>::compute_sort(ps128, ps256);
	merger_t<512>::compute_sort(ps256, outs);
	write_result(out, outs);
}
