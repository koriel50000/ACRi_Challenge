#include "kernel.hpp"
#include <ap_int.h>
#include <multimediaIps/xf_video_mem.hpp>

const int WIDTH = 24;
const int HEIGHT = 24;
const int CHANNEL = 16;

const int KERNEL_SIZE = 2;
const int OWIDTH = WIDTH / KERNEL_SIZE;
const int OHEIGHT = HEIGHT / KERNEL_SIZE;

typedef ap_uint<2> uint2_t;
typedef xf::cv::LineBuffer<KERNEL_SIZE - 1, WIDTH, uint2_t> linebuf_t;
typedef xf::cv::Window<KERNEL_SIZE, KERNEL_SIZE, uint2_t> window_t;

void maxpool(const window_t window, uint2_t& out) {
	uint2_t max = 0;
	for (int y = 0; y < KERNEL_SIZE; y++) {
		for (int x = 0; x < KERNEL_SIZE; x++) {
			uint2_t v = window(x, y);
			if (v > max) {
				max = v;
			}
		}
	}
	out = max;
}

void slide_window(linebuf_t& linebuf, window_t& window, const int x, uint2_t& in) {
	uint2_t rows[KERNEL_SIZE];

	window.shift_pixels_left();
	window.insert_pixel(linebuf(0, x), 1, 1);
}

void kernel(int in[HEIGHT * WIDTH * CHANNEL],
	int out[(HEIGHT / KERNEL_SIZE) * (WIDTH / KERNEL_SIZE) * CHENNEL])
{
#pragma HLS interfalce axis port=in
#pragma HLS interfalce axis port=out
#pragma HLS array_pertition variable=in cyclic factor=WIDTH
#pragma HLS array_pertition variable=out cyclic factor=WIDTH / KERNEL_SIZE

	int iptr = 0;
	int optr = 0;
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS unroll
			linebuf.insert_bottom_row(in[iptr++], x);
		}

		for (int x = 0; x < WIDTH; x += KERNEL_SIZE) {
			for (int k = 0; k < KERNEL_SIZE; k++) {
#pragma HLS unroll
				slide_window(linbuf, window, x + k, in[iptr++]);
			}
			maxpool(window, out[optr++]);
		}
	}
}
