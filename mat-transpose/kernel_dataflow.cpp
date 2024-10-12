#include "kernel.hpp"
#include <hls_stream.h>

typedef hls::stream<float> fifo_t;

void read_input(const float in[SIZE * SIZE], fifo_t& ins) {
	for (int i = 0; i < SIZE * SIZE; i++) {
		ins << in[i];
	}
}

void compute_transpose(fifo_t& ins, fifo_t& outs) {
	float buf[SIZE * SIZE];

	for (int j = 0; j < SIZE - 1; j++) {
#pragma HLS pipeline
		ins >> buf[SIZE * j + 0];
		outs << buf[SIZE * j + 0];
		for (int i = 1; i < SIZE; i++) {
			ins >> buf[SIZE * j + i];
		}
	}
	ins >> buf[SIZE * (SIZE - 1) + 0];
	outs << buf[SIZE * (SIZE - 1) + 0];
	for (int i = 1; i < SIZE; i++) {
#pragma HLS pipeline
		ins >> buf[SIZE * (SIZE - 1) + i];
		for (int j = 0; j < SIZE; j++) {
			outs << buf[SIZE * j + i];
		}
	}
}

void write_result(float out[SIZE * SIZE], fifo_t& outs) {
	for (int i = 0; i < SIZE * SIZE; i++) {
		 outs >> out[i];
	}
}

void kernel(const float in[SIZE * SIZE], float out[SIZE * SIZE]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=64

	fifo_t ins("input_fifo");
	fifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input(in, ins);
	compute_transpose(ins, outs);
	write_result(out, outs);
}
