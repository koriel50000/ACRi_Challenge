#include "kernel.hpp"
#include <hls_stream.h>
#include <hls_vector.h>

const int SIZE = 1024;
const int DEGREE = 128;
const int NUM = 8;

using chunk_t = hls::vector<float, DEGREE>;
template <typename T>
using fifo = hls::stream<T>;

// N = 2, 4, 8, 16, ...
template <typename T, int N>
T quick_sum(const hls::vector<T, N>& v) {
#pragma HLS pipeline
	const int p = 7; //ilogb(N);
	T t[N];
#pragma HLS array_partition variable=t

	for (int i = 0; i < N; i++) {
#pragma HLS unroll
		t[i] = v[i];
	}
	for (int j = 0, d = 1; j < p; j++, d *= 2) {
		for (int i = 0; i + d < N; i += d * 2) {
#pragma HLS unroll
			t[i] += t[i + d];
		}
	}
	return t[0];
}

void read_input(const float in[SIZE], const int size, fifo<chunk_t>& ins) {
	for (int i = 0; i < SIZE; i += DEGREE) {
#pragma HLS pipeline
		chunk_t buf;
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			buf[j] = (i + j < size) ? in[i + j] : 0;
		}
		ins.write(buf);
	}
}

void compute_sum(fifo<chunk_t>& ins, fifo<float>& outs) {
	for (int i = 0; i < NUM; i++) {
#pragma HLS pipeline
		chunk_t buf = ins.read();
		outs.write(quick_sum<float, DEGREE>(buf));
	}
}

void write_result(float *out, fifo<float>& outs) {
	static float acc[NUM];
#pragma HLS array_partition variable=acc

	for (int i = 0; i < NUM; i++) {
#pragma HLS unroll
		acc[i] = outs.read();
	}
	*out = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
		+ ((acc[4] + acc[5]) + (acc[6] + acc[7]));
}

void kernel(const float in[SIZE], const int size, float *out) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=size
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE

	fifo<chunk_t> ins("input_fifo");
	fifo<float> outs("output_fifo");

#pragma HLS dataflow
	read_input(in, size, ins);
	compute_sum(ins, outs);
	write_result(out, outs);
}
