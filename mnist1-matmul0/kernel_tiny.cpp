#include "kernel.hpp"
#include <hls_stream.h>
#include <ap_int.h>

const int AFFINE_SIZE = 256;
const int CLASSES = 10;

const int CHUNK_SIZE = 16;

typedef ap_int<4> int4_t;
typedef ap_uint<64> pack_t;
typedef hls::stream<pack_t> fifo_t;

void read_input(const int weight[CLASSES * AFFINE_SIZE], fifo_t& ins) {
	int ptr = 0;
	for (int j = 0; j < CLASSES; j++) {
#pragma HLS pipeline
		for (int i = 0; i < AFFINE_SIZE / CHUNK_SIZE; i++) {
			int4_t w0  = weight[ptr++];
			int4_t w1  = weight[ptr++];
			int4_t w2  = weight[ptr++];
			int4_t w3  = weight[ptr++];
			int4_t w4  = weight[ptr++];
			int4_t w5  = weight[ptr++];
			int4_t w6  = weight[ptr++];
			int4_t w7  = weight[ptr++];
			int4_t w8  = weight[ptr++];
			int4_t w9  = weight[ptr++];
			int4_t w10 = weight[ptr++];
			int4_t w11 = weight[ptr++];
			int4_t w12 = weight[ptr++];
			int4_t w13 = weight[ptr++];
			int4_t w14 = weight[ptr++];
			int4_t w15 = weight[ptr++];
			pack_t val = (w0 , w1 , w2 , w3 , w4 , w5 , w6 , w7,
				w8 , w9 , w10 , w11 , w12 , w13 , w14 , w15);
			ins.write(val);
		}
	}
}

void compute_affine(const int in[AFFINE_SIZE], fifo_t& ins, fifo_t& outs) {
	int4_t vec[AFFINE_SIZE];
#pragma HLS array_partition variable=vec cyclic factor=CHUNK_SIZE

	for (int i = 0; i < AFFINE_SIZE; i++) {
#pragma HLS unroll factor=CHUNK_SIZE
		vec[i] = in[i];
	}

	for (int j = 0; j < CLASSES; j++) {
#pragma HLS pipeline
		for (int i = 0; i < AFFINE_SIZE; i += CHUNK_SIZE) {
			pack_t v = ins.read();
			int4_t w0  = v.range(63, 60) * vec[i + 0];
			int4_t w1  = v.range(59, 56) * vec[i + 1];
			int4_t w2  = v.range(55, 52) * vec[i + 2];
			int4_t w3  = v.range(51, 48) * vec[i + 3];
			int4_t w4  = v.range(47, 44) * vec[i + 4];
			int4_t w5  = v.range(43, 40) * vec[i + 5];
			int4_t w6  = v.range(39, 36) * vec[i + 6];
			int4_t w7  = v.range(35, 32) * vec[i + 7];
			int4_t w8  = v.range(31, 28) * vec[i + 8];
			int4_t w9  = v.range(27, 24) * vec[i + 9];
			int4_t w10 = v.range(23, 20) * vec[i + 10];
			int4_t w11 = v.range(19, 16) * vec[i + 11];
			int4_t w12 = v.range(15, 12) * vec[i + 12];
			int4_t w13 = v.range(11,  8) * vec[i + 13];
			int4_t w14 = v.range( 7,  4) * vec[i + 14];
			int4_t w15 = v.range( 3,  0) * vec[i + 15];
			pack_t val = (w0 , w1 , w2 , w3 , w4 , w5 , w6 , w7,
				w8 , w9 , w10 , w11 , w12 , w13 , w14 , w15);
			outs.write(val);
		}
        }
}

void write_result(int out[CLASSES], fifo_t& outs) {
	for (int j = 0; j < CLASSES; j++) {
#pragma HLS pipeline
		int acc = 0;
		for (int i = 0; i < AFFINE_SIZE / CHUNK_SIZE; i++) {
			pack_t v = outs.read();
			int4_t w0  = v.range(63, 60);
			int4_t w1  = v.range(59, 56);
			int4_t w2  = v.range(55, 52);
			int4_t w3  = v.range(51, 48);
			int4_t w4  = v.range(47, 44);
			int4_t w5  = v.range(43, 40);
			int4_t w6  = v.range(39, 36);
			int4_t w7  = v.range(35, 32);
			int4_t w8  = v.range(31, 28);
			int4_t w9  = v.range(27, 24);
			int4_t w10 = v.range(23, 20);
			int4_t w11 = v.range(19, 16);
			int4_t w12 = v.range(15, 12);
			int4_t w13 = v.range(11,  8);
			int4_t w14 = v.range( 7,  4);
			int4_t w15 = v.range( 3,  0);
			acc += (((w0 + w1) + (w2 + w3)) +
				((w4 + w5) + (w6 + w7))) +
				(((w8 + w9) + (w10 + w11)) +
				((w12 + w13) + (w14 + w15)));
		}
		out[j] = acc;
	}
}

void kernel(int in[256], int weight[10 * 256], int out[10]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=weight
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=weight cyclic factor=CHUNK_SIZE

	fifo_t ins("input_fifo");
	fifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input(weight, ins);
	compute_affine(in, ins, outs);
	write_result(out, outs);
}
