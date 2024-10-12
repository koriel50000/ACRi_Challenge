#include "kernel.hpp"

#include "hls_vector.h"
#include "hls_stream.h"
#include "hls_math.h"

const int DEGREE = 32;

typedef hls::vector<uint16_t, DEGREE> chunk_t;
typedef hls::stream<chunk_t> fifo_t;

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

void read_input(const uint8_t in[8192], const int size, fifo_t& outs) {
	static uint16_t buf[256 * DEGREE];
#pragma HLS array_partition variable=buf cyclic factor=DEGREE

	for (int i = 0; i < 256 * DEGREE; i++) {
#pragma HLS unroll
		buf[i] = 0;
	}
	const int p = ilogb(DEGREE);

	for (int i = 0; i < size; i++) {
#pragma HLS unroll factor=DEGREE
// @thanks https://acri-vhls-challenge.web.app/user/NapoliN/code/55l7K2SXGtrZLkBYMc4N
		buf[(in[i] << p) + (i & (DEGREE - 1))]++;
	}

	for (int i = 0; i < 256 * DEGREE; i += DEGREE) {
#pragma HLS pipeline
		chunk_t chunk;
		for (int k = 0; k < DEGREE; k++) {
#pragma HLS unroll
			chunk[k] = buf[i + k];
		}
		outs << chunk;
	}
}

void write_result(uint16_t hist[256], fifo_t& outs) {
	for (int i = 0; i < 256; i++) {
#pragma HLS pipeline
		chunk_t chunk;
		outs >> chunk;
		hist[i] = quick_sum<uint16_t, DEGREE>(chunk);
	}
}

void kernel(const uint8_t in[8192], const int size, uint16_t hist[256]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=size
#pragma HLS interface axis port=hist
#pragma HLS array_partition variable=in cyclic factor=DEGREE

	fifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input(in, size, outs);
	write_result(hist, outs);
}
