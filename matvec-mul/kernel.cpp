#include "kernel.hpp"
#include <hls_stream.h>
#include <hls_vector.h>

const int CHUNK_SIZE = 32;

using chunk_t = hls::vector<float, CHUNK_SIZE>;
using fifo_t = hls::stream<chunk_t>;

// @see HD, Figure 3-3
constexpr size_t clp2(size_t x) {
	x = x - 1;
	x = x | (x >> 1);
	x = x | (x >> 2);
	x = x | (x >> 4);
	x = x | (x >> 8);
	x = x | (x >> 16);
	return x + 1;
}

template <typename T, int N>
T quick_sum(const hls::vector<T, N>& v) {
	const size_t M = clp2(N);
	T t[M];
#pragma HLS array_partition variable=t

	for (int i = 0; i < N; i++) {
#pragma HLS unroll
		t[i] = v[i];
	}
	for (int i = N; i < M; i++) {
#pragma HLS unroll
		t[i] = 0;
	}

	for (int d = 2; d <= M; d *= 2) {
		for (int i = 0; i < M; i += d) {
#pragma HLS unroll
			t[i] += t[i + d / 2];
		}
	}
	return t[0];
}

void read_input(const float in_mat[SIZE * SIZE], fifo_t& ins) {
	for (int i = 0; i < SIZE * SIZE; i += CHUNK_SIZE) {
#pragma HLS pipeline
		chunk_t chunk;
		for (int k = 0; k < CHUNK_SIZE; k++) {
#pragma HLS unroll
			chunk[k] = in_mat[i + k];
		}
		ins.write(chunk);
	}
}

void compute_mul(const float in_vec[SIZE], fifo_t& ins, fifo_t& outs) { 
	float vec[SIZE];
#pragma HLS array_partition variable=vec cyclic factor=CHUNK_SIZE

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll factor=CHUNK_SIZE
		vec[i] = in_vec[i];
	}

	for (int j = 0; j < SIZE; j++) {
#pragma HLS pipeline
		for (int i = 0; i < SIZE; i += CHUNK_SIZE) {
			chunk_t chunk = ins.read();
			for (int k = 0; k < CHUNK_SIZE; k++) {
#pragma HLS unroll
				chunk[k] *= vec[i + k];
			}
			outs.write(chunk);
		}
	}
}

void write_result(float out[SIZE], fifo_t& outs) {
	for (int j = 0; j < SIZE; j++) {
#pragma HLS pipeline
		float acc = 0;
		for (int i = 0; i < SIZE / CHUNK_SIZE; i++) {
			chunk_t chunk = outs.read();
			acc += quick_sum<float, CHUNK_SIZE>(chunk);
		}
		out[j] = acc;
	}
}

void kernel(const float in_mat[SIZE * SIZE], const float in_vec[SIZE], float out[SIZE]) {
#pragma HLS interface axis port=in_mat
#pragma HLS interface axis port=in_vec
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in_mat cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=in_vec cyclic factor=CHUNK_SIZE

	fifo_t ins("input_fifo");
	fifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input(in_mat, ins);
	compute_mul(in_vec, ins, outs);
	write_result(out, outs);
}
