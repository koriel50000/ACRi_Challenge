#include "kernel.hpp"
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>

using chunk_t = hls::vector<float, SIZE>;
template <typename T>
using fifo = hls::stream<T>;

// @see HD, Figure 3-3
constexpr int clp2(int x) {
	x = x - 1;
	x = x | (x >> 1);
	x = x | (x >> 2);
	x = x | (x >> 4);
	x = x | (x >> 8);
	x = x | (x >> 16);
	return x + 1;
}

// M = 2 * m
template <typename T, int M>
T quick_sum(const hls::vector<T, M>& v) {
	const int N = clp2(M);
	const int p = ilogb(N);
	T t[M];
#pragma HLS array_partition variable=t

	for (int i = 0; i < M; i++) {
#pragma HLS unroll
		t[i] = v[i];
	}
	for (int j = 0, d = 1; j < p; j++, d *= 2) {
		for (int i = 0; i + d < M; i += d * 2) {
#pragma HLS unroll
			t[i] += t[i + d];
		}
	}
	return t[0];
}

void read_input(const float in_mat[SIZE * SIZE], fifo<chunk_t>& ins) {
	for (int i = 0; i < SIZE * SIZE; i += SIZE) {
#pragma HLS pipeline
		chunk_t val;
		for (int j = 0; j < SIZE; j++) {
#pragma HLS unroll
			val[j] = in_mat[i + j];
		}
		ins.write(val);
	}
}

void compute_matmul(const float in_vec[SIZE],
		fifo<chunk_t>& ins, fifo<chunk_t>& outs)
{
	float vec[SIZE];
#pragma HLS array_partition variable=vec

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		vec[i] = in_vec[i];
	}

	for (int i = 0; i < SIZE; i++) {
#pragma HLS pipeline
		chunk_t val = ins.read();
		for (int j = 0; j < SIZE; j++) {
#pragma HLS unroll
			val[j] *= vec[j];
		}
		outs.write(val);
	}
}

void write_result(float out[SIZE], fifo<chunk_t>& outs) {
	for (int i = 0; i < SIZE; i++) {
#pragma HLS pipeline
		chunk_t val = outs.read();
		float acc = quick_sum<float, SIZE>(val);
		out[i] = acc;
	}
}

void kernel(
  const float in_mat[SIZE * SIZE],
  const float in_vec[SIZE],
  float out[SIZE]
) {
#pragma HLS interface axis port=in_mat
#pragma HLS interface axis port=in_vec
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in_mat cyclic factor=SIZE
#pragma HLS array_partition variable=in_vec
#pragma HLS array_partition variable=out

	fifo<chunk_t> ins("input_fifo");
	fifo<chunk_t> outs("output_fifo");

#pragma HLS dataflow
	read_input(in_mat, ins);
	compute_matmul(in_vec, ins, outs);
	write_result(out, outs);
}
