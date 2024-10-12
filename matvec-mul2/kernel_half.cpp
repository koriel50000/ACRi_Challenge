#include "kernel.hpp"
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>

const int HALF = 40;
const int CHUNK = SIZE + HALF;

using chunk_t = hls::vector<float, CHUNK>;
using half_t = hls::vector<float, HALF>;
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

template <typename T, int M>
T quick_sum(T *v0, T *v1) {
	//const int N = clp2(M);
	const int p = 6; //ilogb(N);
	T t[M];
#pragma HLS array_partition variable=t

	for (int i = 0; i < M; i++) {
#pragma HLS unroll
		t[i] = v0[i] + v1[i];
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
	for (int i = 0; i < SIZE * SIZE + CHUNK - 1; i += CHUNK) {
#pragma HLS pipeline
		chunk_t val;
		for (int j = 0; j < CHUNK; j++) {
#pragma HLS unroll
			val[j] = (i + j < SIZE * SIZE) ? in_mat[i + j] : 0;
		}
		ins.write(val);
	}
}

void compute_matmul(const float in_vec[SIZE],
		fifo<chunk_t>& ins, fifo<chunk_t>& outs)
{
	float vec[SIZE + SIZE];
#pragma HLS array_partition variable=vec

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		float val = in_vec[i];
		vec[i] = val;
		vec[SIZE + i] = val;
	}

	for (int i = 0; i < (SIZE * SIZE + CHUNK - 1) / CHUNK; i++) {
#pragma HLS pipeline
		chunk_t mat = ins.read();
		int ptr = (i & 1) * HALF;
		for (int j = 0; j < CHUNK; j++) {
#pragma HLS unroll
			mat[j] *= vec[ptr++];
		}
		outs.write(mat);
	}
}

void write_result(float out[SIZE], fifo<chunk_t>& outs) {
	float v0[CHUNK];
	float v1[CHUNK];
#pragma HLS array_partition variable=v0
#pragma HLS array_partition variable=v1

	for (int i = 0; i < SIZE; i += 3) {
#pragma HLS pipeline
		const chunk_t val0 = outs.read();
		for (int j = 0; j < CHUNK; j++) {
#pragma HLS unroll
			v0[j] = val0[j];
		}
		out[i + 0] = quick_sum<float, HALF>(&v0[0], &v0[HALF]);
		const chunk_t val1 = outs.read();
		for (int j = 0; j < CHUNK; j++) {
#pragma HLS unroll
			v1[j] = val1[j];
		}
		out[i + 1] = quick_sum<float, HALF>(&v0[SIZE], &v1[0]);
		if (i + 2 == SIZE) break;
		out[i + 2] = quick_sum<float, HALF>(&v1[HALF], &v1[SIZE]);
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
#pragma HLS array_partition variable=in_mat cyclic factor=CHUNK
#pragma HLS array_partition variable=in_vec
#pragma HLS array_partition variable=out

	fifo<chunk_t> ins("input_fifo");
	fifo<chunk_t> outs("output_fifo");

#pragma HLS dataflow
	read_input(in_mat, ins);
	compute_matmul(in_vec, ins, outs);
	write_result(out, outs);
}
