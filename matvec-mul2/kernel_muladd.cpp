#include "kernel.hpp"
#include <hls_stream.h>
#include <hls_vector.h>

const int HALF = SIZE / 2;
const int CHUNK = SIZE + HALF;
const int COUNT = (SIZE * SIZE + CHUNK - 1) / CHUNK;

using chunk_t = hls::vector<float, CHUNK>;
template <typename T>
using fifo = hls::stream<T>;

template <typename T>
T muladd(T mat[SIZE * 2], T vec[SIZE]) {
#pragma HLS inline
	//const int N = clp2(M);
	const int p = 7; //ilogb(N);
	T t[SIZE];
#pragma HLS array_partition variable=t

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		t[i] = mat[i] * vec[i];
	}

	for (int j = 0, d = 1; j < p; j++, d *= 2) {
		for (int i = 0; i + d < SIZE; i += d * 2) {
#pragma HLS unroll
			t[i] += t[i + d];
		}
	}
	return t[0];
}

void read_input(const float in_mat[SIZE * SIZE], fifo<chunk_t>& ins) {
	int ptr = 0;
	for (int i = 0; i < COUNT; i++) {
#pragma HLS pipeline
		chunk_t val;
		for (int j = 0; j < CHUNK; j++) {
#pragma HLS unroll
			val[j] = (ptr < SIZE * SIZE) ? in_mat[ptr++] : 0;
		}
		ins.write(val);
	}
}

int offset(unsigned int i) {
	static const int table[] {
		0, 40, -1, 0, 40, -1, 0, 40, -1,
		0, 40, -1, 0, 40, -1, 0, 40, -1,
		0, 40, -1, 0, 40, -1, 0, 40, -1,
		0, 40, -1, 0, 40, -1, 0, 40, -1,
		0, 40, -1, 0, 40, -1, 0, 40, -1,
		0, 40, -1, 0, 40, -1, 0, 40, -1,
		0, 40, -1, 0, 40, -1, 0, 40, -1,
		0, 40, -1, 0, 40, -1, 0, 40, -1,
		0, 40, -1, 0, 40, -1, 0, 40,
	};
	return table[i];
}

void lshift(float buf[SIZE * 2]) {
#pragma HLS inline
	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		buf[i] = buf[i + SIZE];
	}
}

void compute_result(const float in_vec[SIZE],
		float out[SIZE], fifo<chunk_t>& ins)
{
	float vec[SIZE];
#pragma HLS array_partition variable=vec

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		vec[i] = in_vec[i];
	}

	float buf[SIZE * 2];
#pragma HLS array_partition variable=buf

	for (int i = 0; i < SIZE; i++) {
#pragma HLS pipeline
		lshift(buf);
		int o = offset(i);
		if (o >= 0) {
			const chunk_t val = ins.read();
			for (int j = 0; j < CHUNK; j++) {
#pragma HLS unroll
				buf[o + j] = val[j];
			}
		}
		out[i] = muladd<float>(buf, vec);
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

#pragma HLS dataflow
	read_input(in_mat, ins);
	compute_result(in_vec, out, ins);
}
