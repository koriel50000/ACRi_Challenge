#include "kernel.hpp"
#include <hls_stream.h>

typedef struct {
	float x;
	float y;
} coord_t;

typedef hls::stream<coord_t> ififo_t;
typedef hls::stream<bool> ofifo_t;

bool divergence_value(const int max_iter, const float a, const float b) {
	float nx = 0.0f;
	float ny = 0.0f;
	for (int i = 0; i < max_iter; i++) {
		float x2 = nx * nx;
		float y2 = ny * ny;
		float xy = nx * ny * 2.0f;
		if (x2 + y2 > 4.0f) {
			return false;
		}
		nx = x2 - y2 + a;
		ny = xy + b;
	}
	return true;
}

void create_input(const int width, const int height, const float x0, const float y0, const float dx, const float dy, ififo_t& ins) {
	float y = y0;
	for (int j = 0; j < height; j++, y += dy) {
		float x = x0;
		for (int i = 0; i < width; i++, x += dx) {
			coord_t coord;
			coord.x = x;
			coord.y = y;
			ins.write(coord);
		}
	}
}

void compute_mandelbrot(const int size, const int max_iter, ififo_t& ins, ofifo_t& outs) {
	for (int j = 0; j < size; j++) {
		coord_t coord = ins.read();
		bool plot = divergence_value(max_iter, coord.x, coord.y);
		outs.write(plot);
	}
}

void write_result(const int size, const int max_iter, char output[MAX_OUTPUT_SIZE], ofifo_t& outs) {
	for (int j = 0; j < size; j++) {
		bool plot = outs.read();
		output[j] = plot ? '*' : ' ';
	}
}

// @see https://ja.wikipedia.org/wiki/%E3%83%9E%E3%83%B3%E3%83%87%E3%83%AB%E3%83%96%E3%83%AD%E9%9B%86%E5%90%88
// x[n+1] = x[n]^2 - y[n]^2 + a
// y[n+1] = 2 * x[n] * y[n] + b
void kernel(
  const int width,
  const int height,
  const int max_iter,
  const float start_x,
  const float start_y,
  const float step_x,
  const float step_y,
  char output[MAX_OUTPUT_SIZE]
) {
#pragma HLS interface axis port=width
#pragma HLS interface axis port=height
#pragma HLS interface axis port=max_iter
#pragma HLS interface axis port=start_x
#pragma HLS interface axis port=start_y
#pragma HLS interface axis port=step_x
#pragma HLS interface axis port=step_y
#pragma HLS interface axis port=output

	ififo_t ins("input_fifo");
	ofifo_t outs("output_fifo");

	const int size = width * height;

#pragma HLS dataflow
	create_input(width, height, start_x, start_y, step_x, step_y, ins);
	compute_mandelbrot(size, max_iter, ins, outs);
	write_result(size, max_iter, output, outs);
}
