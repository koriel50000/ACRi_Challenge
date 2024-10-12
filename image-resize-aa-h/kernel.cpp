#include "kernel.hpp"
#include <hls_stream.h>
#include <ap_int.h>

typedef ap_uint<11> len_t;
typedef ap_uint<18> sum_t;

const int in_size = 1024;

typedef hls::stream<uint8_t> ififo_t;
typedef hls::stream<sum_t> ofifo_t;

void read_input(const uint8_t in[in_size], ififo_t& ins) {
	for (int i = 0; i < in_size; i++) {
		ins << in[i]; 
	}
}

// @see https://en.wikipedia.org/wiki/Digital_differential_analyzer_(graphics_algorithm)
void compute_resize(const len_t out_size, ififo_t& ins, ofifo_t& outs) {
	sum_t sum = 0;
	len_t d = 0;
	for (int i = 0; i < in_size; i++) {
		uint8_t v;
		ins >> v;
		d += out_size;
		if (d < in_size) {
			sum += v * out_size;
		} else {
			d -= in_size;
			sum += v * (out_size - d);
			outs << sum;
			sum = v * d;
		}
	}
}

void write_result(len_t out_size, uint8_t out[1024], ofifo_t& outs) {
	for (int i = 0; i < out_size; i++) {
		sum_t sum;
		outs >> sum;
		out[i] = (sum + in_size / 2) / in_size;
	}
}

void kernel(const uint8_t in[1024],
	const uint32_t out_size, uint8_t out[1024])
{
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS interface axis port=out_size

	ififo_t ins("input_fifo");
	ofifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input(in, ins);
	compute_resize(out_size, ins, outs);
	write_result(out_size, out, outs);
}
