#include "kernel.hpp"

const int DEGREE = 32;

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

	float lookup[256];
#pragma HLS array_partition variable=lookup

	for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor=DEGREE
		lookup[i] = table[i];
	}
	
	for (int i = 0; i < size; i++) {
#pragma HLS unroll factor=DEGREE
		out[i] = lookup[in[i]];
	}
}
