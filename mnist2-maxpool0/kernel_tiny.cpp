#include "kernel.hpp"
#include <ap_int.h>

const int WIDTH = 24;
const int HEIGHT = 24;
const int CHANNEL = 16;

const int KERNEL_SIZE = 2;
const int OWIDTH = WIDTH / KERNEL_SIZE;
const int OHEIGHT = HEIGHT / KERNEL_SIZE;

typedef ap_uint<2> uint2_t;
typedef ap_uint<CHANNEL * 2> pack_t;

void maxpool(pack_t& val1, const pack_t val2) {
	pack_t val = window[i];
	for (int z = 0; z < CHANNEL; z++) {
#pragma HLS unroll
		int p = z * 2;
		uint2_t v1 = val(p + 1, p);
		uint2_t v2 = max(p + 1, p);
		if (v2 > v1) {
			val1(p + 1, p) = v2;
		}
	}
}

void read_pack(const int in[HEIGHT * WIDTH * CHANNEL], int& ptr, pack_t& val) {
	for (int i = 0; i < CHANNEL; i++) {
#pragma HLS unroll
		int p = i * 2;
		uint2_t v = in[ptr++];
		val(p + 1, p) = v;
	}
}

void write_pack(int out[HEIGHT * WIDTH * CHANNEL], int& ptr, const pack_t& val) {
	for (int i = 0; i < CHANNEL; i++) {
#pragma HLS unroll
		int p = i * 2;
		uint2_t v = val(p + 1, p);
		out[ptr++] = v;
	}
}

void kernel(int in[HEIGHT * WIDTH * CHANNEL],
	int out[OHEIGHT * OWIDTH * CHANNEL])
{
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHANNEL * KERNEL_SIZE
#pragma HLS array_partition variable=out cyclic factor=CHANNEL

	pack_t linebuf[OWIDTH];
#pragma HLS array_partition variable=linebuf

	int iptr = 0;
	int optr = 0;
	for (int y = 0; y < HEIGHT; y += 2) {
#pragma HLS pipeline
		for (int x = 0; x < OWIDTH; x++) {
#pragma HLS unroll
			pack_t val1;
			read_pack(in, iptr, val1);
			pack_t val2;
			read_pack(in, iptr, val2);
			maxpool(val1, val2);
			linebuf[x] = val1;
		}
		for (int x = 0; x < OWIDTH; x += 2) {
#pragma HLS unroll
			 = linebuf[x];
			window[1] = linebuf[x + 1];
			read_pack(in, iptr, window[2]);
			read_pack(in, iptr, window[3]);
			pack_t val = maxpool(window);
			write_pack(out, optr, val);
		}
	}
}
