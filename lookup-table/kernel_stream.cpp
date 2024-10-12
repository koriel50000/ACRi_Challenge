#include "kernel.hpp"
#include <hls_stream.h>

const int DEGREE = 32;

typedef hls::stream<uint8_t> fifo_t;

void read_input(const uint8_t in[1024], const int size, fifo_t& ins) {
	for (int i = 0; i < size; i++) {
#pragma HLS unroll factor=DEGREE
		ins.write(in[i]);
	}
}

void write_result(const float table[256], const int size, float out[1024], fifo_t& ins) {
	float lookup[256];
#pragma HLS array_partition variable=lookup

	for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor=DEGREE
		lookup[i] = table[i];
	}
	
	for (int i = 0; i < size; i++) {
#pragma HLS unroll factor=DEGREE
		uint8_t v = ins.read();
		out[i] = lookup[v];
	}
}

void kernel(
  const float table[256],
  const uint8_t in[1024],
  const int size,
  float out[1024]
) {
#pragma HLS interface axis port=table
#pragma HLS interface axis port=in
#pragma HLS interface axis port=size
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=table cyclic factor=DEGREE
#pragma HLS array_partition variable=in cyclic factor=DEGREE
#pragma HLS array_partition variable=out cyclic factor=DEGREE

	fifo_t ins("input_fifo");

#pragma HLS dataflow
	read_input(in, size, ins);
	write_result(table, size, out, ins);
}
