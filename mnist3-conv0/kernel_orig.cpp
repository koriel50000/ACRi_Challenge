#include "kernel.hpp"
#include <hls_stream.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <assert.h>
//#include <multimediaIps/xf_video_mem.hpp>

const int WIDTH = 28;
const int HEIGHT = 28;
const int CHANNEL = 1;

const int FILTER = 16;
const int KERNEL = 5;
const int THRESHOLD = 3;

const int OWIDTH = WIDTH - KERNEL + 1;
const int OHEIGHT = HEIGHT - KERNEL + 1;

typedef ap_int<2> int2_t;
typedef ap_uint<KERNEL * KERNEL * 2 + 6> pack_t; // 8の倍数にしておく
typedef hls::stream<pack_t> fifo_t;

template <int ROWS, int COLS, typename T>
class LineBuffer {
private:
	hls::vector<T, ROWS * COLS> buf;
public:
	void shift_pixels_up(int col) {
		for (int i = 0; i < ROWS * COLS - 1; i++) {
#pragma HLS unroll
			buf[i] = buf[i + 1];
		}
	}

	void insert_bottom_row(T value, int col) {
		buf[ROWS * COLS - 1] = value;
	}

	void get_col(T value[ROWS], int col) {
		for (int i = 0; i < ROWS; i++) {
#pragma HLS unroll
			value[i] = buf[i * COLS];
		}
	}
};

template <int ROWS, int COLS, typename T>
class Window {
private:
	hls::vector<T, ROWS * COLS> buf;
public:
	void shift_pixels_left() {
		for (int i = 0; i < ROWS * COLS - 1; i++) {
#pragma HLS unroll
			buf[i] = buf[i + 1];
		}
	}

	void insert_right_col(T value[ROWS]) {
		for (int i = 0; i < ROWS; i++) {
#pragma HLS unroll
			buf[(i + 1) * COLS - 1] = value[i];
		}
	}

	T& operator ()(int row, int col) {
		return buf[row * COLS + col];
	}
};

//typedef xf::cv::LineBuffer<KERNEL - 1, WIDTH, int2_t> linebuf_t;
//typedef xf::cv::Window<KERNEL, KERNEL, int2_t> window_t;
typedef LineBuffer<KERNEL - 1, WIDTH, int2_t> linebuf_t;
typedef Window<KERNEL, KERNEL, int2_t> window_t;

void insert_linebuf(linebuf_t& linebuf, const int x, const int2_t v) {
	linebuf.shift_pixels_up(x);
	linebuf.insert_bottom_row(v, x);
}

template<int KH, int KW>
void slide_window(linebuf_t& linebuf, window_t& window, const int x, const int2_t v) {
	int2_t rows[KH];

	linebuf.get_col(rows, x);
	rows[KH - 1] = v;
	insert_linebuf(linebuf, x, v);

	window.shift_pixels_left();
	window.insert_right_col(rows);
}

template<int KH, int KW>
void pack_window(window_t& window, pack_t& val) {
	for (int y = 0; y < KH; y++) {
#pragma HLS unroll
		for (int x = 0; x < KW; x++) {
			int2_t v = window(y, x);
			int p = (y * KW + x) * 2;
			val(p + 1, p) = v;
		}
	}
}

template<int KH, int KW>
void sign_extend(pack_t& val, int width, int msb) {
	for (int p = 0; p < KH * KW * 2; p += width) {
#pragma HLS unroll
		val[p + msb] = val[p + msb - 1];
	}
}

template<int KH, int KW>
ap_int<6> quick_sum(pack_t val) {
	pack_t v1 = (val >> 2 & 0x00333333333333);
	sign_extend<KH, KW>(v1, 4, 2);
	pack_t v2 = (val & 0x03333333333333);
	sign_extend<KH, KW>(v2, 4, 2);
	val = v1 + v2;
	sign_extend<KH, KW>(val, 4, 3);
	val = (val >> 4 & 0x07070707070707) + (val & 0x07070707070707);
	sign_extend<KH, KW>(val, 8, 3);
	val = (val >> 8 & 0x0f000f000f000f) + (val & 0x0f000f000f000f);
	sign_extend<KH, KW>(val, 16, 4);
	val = (val >> 16 & 0x00001f0000001f) + (val & 0x00001f0000001f);
	sign_extend<KH, KW>(val, 32, 5);
	val = (val >> 32 & 0x0000000000003f) + (val & 0x0000000000003f);
	return val(5, 0);
}

template<int H, int W, int C, int KH, int KW>
void read_input(const int in[H * W * C], fifo_t& ins) {
	linebuf_t linebuf;
	window_t window;

	int ptr = 0;
	for (int y = 0; y < KH - 1; y++) {
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
			ap_int<1> v = in[ptr++];
			insert_linebuf(linebuf, x, v);
		}
	}
	for (int y = KH - 1; y < H; y++) {
		for (int x = 0; x < KW - 1; x++) {
#pragma HLS pipeline
			ap_int<1> v = in[ptr++];
			slide_window<KH, KW>(linebuf, window, x, v);
		}
		for (int x = KW - 1; x < W; x++) {
#pragma HLS pipeline
			ap_int<1> v = in[ptr++];
			slide_window<KH, KW>(linebuf, window, x, v);
			pack_t val;
			pack_window<KH, KW>(window, val);
			ins.write(val);
		}
	}
}

template<int F, int KH, int KW, int C, int OH, int OW>
void compute_conv2d(
	const int weight[F * KH * KW * C],
	fifo_t& ins, fifo_t& outs)
{
	pack_t filters[F];
#pragma HLS array_partition variable=filters

	int ptr = 0;
	for (int w = 0; w < F; w++) {
#pragma HLS pipeline
		for (int k = 0; k < KH * KW; k++) {
#pragma HLS unroll
			int2_t v = weight[ptr++];
			int p = k * 2;
			filters[w](p + 1, p) = v;
		}
	}

	for (int i = 0; i < OH * OW; i++) {
#pragma HLS pipeline
		pack_t val = ins.read();
		for (int w = 0; w < F; w++) {
#pragma HLS unroll
			pack_t v = val & filters[w];
			outs.write(v);
		}
	}
}

template<int M, int F, int KH, int KW, int OH, int OW>
void write_result(
	const int threshold[M],
	int out[OH * OW * F], fifo_t& outs)
{
	ap_int<6> thr[M];
#pragma HLS array_partition variable=thr

	for (int n = 0; n < M; n++) {
#pragma HLS unroll
		thr[n] = threshold[n];
	}

	int ptr = 0;
	for (int i = 0; i < OH * OW; i++) {
		for (int j = 0; j < F; j++) {
#pragma HLS pipeline
			pack_t val = outs.read();
			ap_int<6> sum = quick_sum<KH, KW>(val);
			int m = 0;
			for (int n = 0; n < M; n++) {
#pragma HLS unroll
				if (sum >= threshold[n]) {
					m = n + 1;
				}
			}
			out[ptr++] = m;
		}
	}
}

void kernel(
	int in[HEIGHT * WIDTH * CHANNEL],
	int weight[FILTER * KERNEL * KERNEL * CHANNEL],
	int threshold[THRESHOLD],
	int out[OHEIGHT * OWIDTH * FILTER])
{
#pragma HLS interface axis port=in
#pragma HLS interface axis port=weight
#pragma HLS interface axis port=threshold
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=WIDTH
#pragma HLS array_partition variable=weight cyclic factor=KERNEL * KERNEL

	fifo_t ins("input_fifo");
	fifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input<28, 28, 1, 5, 5>(in, ins);
	compute_conv2d<16, 5, 5, 16, 24, 24>(weight, ins, outs);
	write_result<3, 16, 5, 5, 24, 24>(threshold, out, outs);
}
