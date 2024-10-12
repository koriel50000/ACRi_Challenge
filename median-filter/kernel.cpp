#include "kernel.hpp"
#include <hls_stream.h>
#include <assert.h>
#include <multimediaIps/xf_video_mem.hpp>
// @see https://xilinx.github.io/Vitis_Libraries/vision/2021.1/index.html

// @see https://marsee101.blog.fc2.com/blog-entry-3418.html
// @see https://github.com/hashi0203/Vitis_HLS_Gaussian/blob/main/src/gaussian.cpp
// xf::cv::LineBuffer, xf::cv::Window
// @see https://xilinx.github.io/Vitis_Libraries/vision/2021.1/overview.html#xf-cv-linebuffer
// @see https://xilinx.github.io/Vitis_Libraries/vision/2021.1/overview.html#xf-cv-window
typedef xf::cv::LineBuffer<2, WIDTH, uint8_t> linebuf_t;
typedef xf::cv::Window<3, 3, uint8_t> window_t;

typedef hls::stream<uint8_t> fifo_t;

void insert_sort2(uint8_t rank[2], const uint8_t v) {
	bool b0 = (v > rank[0]);
	bool b1 = (v > rank[1]);
	if (b0) {
		rank[1] = rank[0];
		rank[0] = v;
	} else if (b1) {
		rank[1] = v;
	}
}

void insert_sort3(uint8_t rank[3], const uint8_t v) {
	bool b0 = (v > rank[0]);
	bool b1 = (v > rank[1]);
	bool b2 = (v > rank[2]);
	if (b0) {
		rank[2] = rank[1];
		rank[1] = rank[0];
		rank[0] = v;
	} else if (b1) {
		rank[2] = rank[1];
		rank[1] = v;
	} else if (b2) {
		rank[2] = v;
	}
}

void insert_sort5(uint8_t rank[5], const uint8_t v) {
	bool b0 = (v > rank[0]);
	bool b1 = (v > rank[1]);
	bool b2 = (v > rank[2]);
	bool b3 = (v > rank[3]);
	bool b4 = (v > rank[4]);
	if (b0) {
		rank[4] = rank[3];
		rank[3] = rank[2];
		rank[2] = rank[1];
		rank[1] = rank[0];
		rank[0] = v;
	} else if (b1) {
		rank[4] = rank[3];
		rank[3] = rank[2];
		rank[2] = rank[1];
		rank[1] = v;
	} else if (b2) {
		rank[4] = rank[3];
		rank[3] = rank[2];
		rank[2] = v;
	} else if (b3) {
		rank[4] = rank[3];
		rank[3] = v;
	} else if (b4) {
		rank[4] = v;
	}
}

uint8_t corner_median(const uint8_t v[4]) {
	uint8_t rank[2] = { 0, 0 };
	for (int i = 0; i < 4; i++) {
		insert_sort2(rank, v[i]);
	}
	return rank[1];
}

uint8_t edge_median(const uint8_t v[6]) {
	uint8_t rank[3] =  { 0, 0, 0 };
	for (int i = 0; i < 6; i++) {
		insert_sort3(rank, v[i]);
	}
	return rank[2];
}

uint8_t inside_median(const uint8_t v[9]) {
	uint8_t rank[5] = { 0, 0, 0, 0, 0 };
	for (int i = 0; i < 9; i++) {
		insert_sort5(rank, v[i]);
	}
	return rank[4];
}

void slide_window(linebuf_t& linebuf, window_t& window, const int x, fifo_t& ins) {
	uint8_t rows[3];

	linebuf.get_col(rows, x);
	rows[2] = ins.read();
	linebuf.shift_pixels_up(x);
	linebuf.insert_bottom_row(rows[2], x);

	window.shift_pixels_left();
	window.insert_right_col(rows);
}

void slide_window(linebuf_t& linebuf, window_t& window, const int x) {
	window.shift_pixels_left();
	window.insert_pixel(linebuf(0, x), 1, 2);
	window.insert_pixel(linebuf(1, x), 2, 2);
}

void output_corner(window_t& window, fifo_t& outs) {
	uint8_t t[] = {
		window(1, 1), window(1, 2),
		window(2, 1), window(2, 2) 
	};
	outs.write(corner_median(t));
}

void output_h_edge(window_t& window, fifo_t& outs) {
	uint8_t t[] = {
		window(1, 0), window(1, 1), window(1, 2),
		window(2, 0), window(2, 1), window(2, 2)
	};
	outs.write(edge_median(t));
}

void output_v_edge(window_t& window, fifo_t& outs) {
	uint8_t t[] = {
		window(0, 1), window(0, 2),
		window(1, 1), window(1, 2),
		window(2, 1), window(2, 2) 
	};
	outs.write(edge_median(t));
}

void output_inside(window_t& window, fifo_t& outs) {
	uint8_t t[] = {
		window(0, 0), window(0, 1), window(0, 2),
		window(1, 0), window(1, 1), window(1, 2),
		window(2, 0), window(2, 1), window(2, 2)
	};
	outs.write(inside_median(t));
}

void top(linebuf_t& linebuf, window_t& window, fifo_t& ins, fifo_t& outs) {
	for (int x = 0; x < WIDTH + 1; x++) {
		if (x < WIDTH) {
			slide_window(linebuf, window, x, ins);
		}

		if (x == 1 || x == WIDTH) {
			output_corner(window, outs);
		} else if (x > 0) {
			output_h_edge(window, outs);
		}
	}
}

void middle(linebuf_t& linebuf, window_t& window, fifo_t& ins, fifo_t& outs) {
	for (int x = 0; x < WIDTH + 1; x++) {
		if (x < WIDTH) {
			slide_window(linebuf, window, x, ins);
		}

		if (x == 1 || x == WIDTH) {
			output_v_edge(window, outs);
		} else if (x > 0) {
			output_inside(window, outs);
		}
	}
}

void bottom(linebuf_t& linebuf, window_t& window, fifo_t& outs) {
	for (int x = 0; x < WIDTH + 1; x++) {
		if (x < WIDTH) {
			slide_window(linebuf, window, x);
		}

		if (x == 1 || x == WIDTH) {
			output_corner(window, outs);
		} else if (x > 0) {
			output_h_edge(window, outs);
		}
	}
}

void preload(linebuf_t& linebuf, fifo_t& ins) {
	for (int x = 0; x < WIDTH; x++) {
		linebuf.insert_bottom_row(ins.read(), x);
	}
}

void read_input(const uint8_t in[WIDTH * HEIGHT], fifo_t& ins) {
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
//#pragma HLS unroll factor=128
		ins.write(in[i]);
	}
}

void compute_median(fifo_t& ins, fifo_t& outs) {
	// @see https://www.acri.c.titech.ac.jp/wordpress/archives/8947
	linebuf_t linebuf;
	window_t window;

	preload(linebuf, ins);
	// line 0, line 1
	top(linebuf, window, ins, outs);

	for (int y = 2; y < HEIGHT; y++) {
		// line y - 1, line y, line y + 1
		middle(linebuf, window, ins, outs);
	}

	// line HEIGHT - 2, line HEIGHT - 1
	bottom(linebuf, window, outs);
}

void write_result(uint8_t out[WIDTH * HEIGHT], fifo_t& outs) {
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
		out[i] = outs.read();
	}
}

void kernel(const uint8_t in[WIDTH * HEIGHT], uint8_t out[WIDTH * HEIGHT]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_pertition variable=in cyclic factor=128

	fifo_t ins("input_fifo");
	fifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input(in, ins);
	compute_median(ins, outs);
	write_result(out, outs);
}
