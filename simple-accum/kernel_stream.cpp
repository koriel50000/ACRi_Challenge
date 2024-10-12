#include "kernel.hpp"
#include "hls_vector.h"
#include "hls_stream.h"
#include "hls_math.h"

const int DEGREE = 32;

typedef hls::stream<float> fifo_t;
typedef hls::vector<float, DEGREE> chunk_t;

// N = 2, 4, 8, 16, ...
template <typename T, int N>
T quick_sum(const hls::vector<T, N>& v) {
	const int p = ilogb(N) - 1;
	T t[N / 2];
#pragma HLS array_partition variable=t

	int d = 1;
	for (int i = 0; i < N / 2; i++) {
#pragma HLS unroll
		t[i] = v[i * 2] + v[i * 2 + 1];
	}
	for (int j = 0; j < p; j++) {
		d *= 2;
		for (int i = 0; i < N / 2; i += d) {
#pragma HLS unroll
			t[i] += t[i + d / 2];
		}
	}
	return t[0];
}

void read_input(const float in[1024], const int size, fifo_t& ins) {
	for (int i = 0; i < size; i++) {
#pragma HLS unroll factor=DEGREE
		ins << in[i];
	}
}

void write_result(const int size, float *out, fifo_t& ins) {
	static chunk_t buf;

	for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
		buf[j] = 0.0f;
	}

	for (int i = 0; i < size; i++) {
#pragma HLS unroll factor=DEGREE
		buf[i & (DEGREE - 1)] += ins.read();
	}

	*out = quick_sum<float, DEGREE>(buf);
}

void kernel(const float in[1024], const int size, float *out) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=size
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE

	fifo_t ins("input_fifo");

#pragma HLS dataflow
	read_input(in, size, ins);
	write_result(size, out, ins);
}
