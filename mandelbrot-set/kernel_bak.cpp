#include "kernel.hpp"

const int DEGREE = 16;

bool do_plot(const int max_iter, const float a, const float b) {
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
#pragma HLS interface axis port=output
#pragma HLS array_partition variable=output cyclic factor=DEGREE
	int ptr = 0;
	for (int y = 0; y < height; y++) {
		float b = start_y + step_y * y;
		for (int x = 0; x < width; x++) {
#pragma HLS unroll factor=DEGREE skip_exit_check
			float a = start_x + step_x * x;
			bool plot = do_plot(max_iter, a, b);
			output[ptr++] = plot ? '*' : ' ';
		}
	}
}
