#include "kernel.hpp"
#include <hls_stream.h>
#include <ap_int.h>

const int WIDTH = 24;
const int HEIGHT = 24;
const int CHANNEL = 16;

const int OWIDTH = WIDTH / 2;
const int OHEIGHT = HEIGHT / 2;

typedef ap_uint<2> uint2_t;
typedef ap_uint<CHANNEL * 2> pack_t;
typedef hls::stream<pack_t> fifo_t;

template<int C>
void maxpool(pack_t& val1, const pack_t val2) {
	for (int z = 0; z < C; z++) {
#pragma HLS unroll
		int p = z * 2;
		uint2_t v1 = val1(p + 1, p);
		uint2_t v2 = val2(p + 1, p);
		if (v2 > v1) {
			val1(p + 1, p) = v2;
		}
	}
}

template<int H, int W, int C>
void read_input(const int in[H * W * C], fifo_t& ins) {
	int ptr = 0;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x += 2) {
#pragma HLS pipeline
			pack_t val1;
			for (int z = 0; z < C; z++) {
#pragma HLS unroll
				int p = z * 2;
				val1(p + 1, p) = in[ptr++];
			}
			pack_t val2;
			for (int z = 0; z < C; z++) {
#pragma HLS unroll
				int p = z * 2;
				val2(p + 1, p) = in[ptr++];
			}
			maxpool<C>(val1, val2);
			ins.write(val1);
		}
	}
}

template<int H, int W, int C>
void compute_maxpool(fifo_t& ins, fifo_t& outs) {
	pack_t buf[W];
#pragma HLS array_partition variable=buf

	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
			pack_t val = ins.read();
			buf[x] = val;
		}
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
			pack_t val1 = buf[x];
			pack_t val2 = ins.read();
			maxpool<C>(val1, val2);
			outs.write(val1);
		}
	}
}

template<int H, int W, int C>
void write_result(int out[H * W * C], fifo_t& outs) {
	int ptr = 0;
	for (int i = 0; i < H * W; i++) {
#pragma HLS pipeline
		pack_t val = outs.read();
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			int p = z * 2;
			uint2_t v = val(p + 1, p);
			out[ptr++] = v;
		}
	}
}

void kernel(int in[HEIGHT * WIDTH * CHANNEL],
	int out[OHEIGHT * OWIDTH * CHANNEL])
{
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHANNEL * 2
#pragma HLS array_partition variable=out cyclic factor=CHANNEL

	fifo_t ins("input_fifo");
	fifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input<24, 24, 16>(in, ins);
	compute_maxpool<12, 12, 16>(ins, outs);
	write_result<12, 12, 16>(out, outs);
}
