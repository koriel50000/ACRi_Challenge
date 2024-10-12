#include "kernel.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

const int WIDTH = 28;
const int HEIGHT = 28;

const int FILTER = 16;
const int KERNEL = 5;
const int THRESHOLD = 3;

const int OWIDTH = WIDTH - KERNEL + 1;
const int OHEIGHT = HEIGHT - KERNEL + 1;

using int2_t = ap_int<2>;
using uint2_t = ap_uint<2>;
using int2x25_t = ap_uint<2 * KERNEL * KERNEL>;
using int2x16_t = ap_uint<2 * FILTER>;
template <typename T>
using fifo = hls::stream<T>;

namespace bit {
	template <int S>
	int2_t get(const ap_uint<2 * S>& src, const int idx) {
#pragma HLS inline
		int p = 2 * idx;
		return src(p + 2 - 1, p);
	}

	template <int S>
	void set(ap_uint<2 * S>& src, const int idx, const int2_t& v) {
#pragma HLS inline
		int p = 2 * idx;
		src(p + 2 - 1, p) = v;
	}

	template <int S>
	uint2_t getu(const ap_uint<2 * S>& src, const int idx) {
#pragma HLS inline
		int p = 2 * idx;
		return src(p + 2 - 1, p);
	}

	// @see HD, Figure 3-3
	constexpr int clp2(int x) {
		x = x - 1;
		x = x | (x >> 1);
		x = x | (x >> 2);
		x = x | (x >> 4);
		x = x | (x >> 8);
		x = x | (x >> 16);
		return x + 1;
	}

	template <int S>
	int16_t multiply_add(ap_uint<2 * S>& vu, ap_uint<2 * S>& wi) {
		const int M = clp2(S);
		int16_t t[M];
#pragma HLS array_partition variable=t

		for (int i = 0; i < S; i++) {
#pragma HLS unroll
			uint2_t v = getu<S>(vu, i);
			int2_t w = get<S>(wi, i);
			t[i] = v * w;
		}
		for (int i = S; i < M; i++) {
#pragma HLS unroll
			t[i] = 0;
		}

		for (int d = 1; d < M; d *= 2) {
			for (int i = 0; i < M; i += d * 2) {
#pragma HLS unroll
				t[i] += t[i + d];
			}
		}
		return t[0];
	}
} // namespace bit

class Window_0 {
private:
	static const int ROWS = 5;
	static const int COLS = 5;

	int2x25_t buf;
public:
	void shift_pixels_left() {
		buf >>= 2;
	}

	void insert_right_col(const int2_t value[ROWS]) {
		for (int i = 0; i < ROWS; i++) {
#pragma HLS unroll
			int idx = (i + 1) * COLS - 1;
			bit::set<ROWS * COLS>(buf, idx, value[i]);
		}
	}

	// xf::cv::Window<ROWS, COLS, T>にI/Fを合わせたかったが
	// 泣く泣くWindowクラスに関数を追加
	// @see ug1399 Virtual Functions and Pointers: Not supported.

	void set_window(const int idx, const int weight[], const int offset) {
		bit::set<ROWS * COLS>(buf, idx, weight[offset]);
	}

	int16_t muladd(Window_0& weight) {
		return bit::multiply_add<ROWS * COLS>(buf, weight.buf);
	}
};

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

template <typename WT, typename T, int H, int W, int C, int KH, int KW, int F, int M>
class Conv2D {
private:
	static const int OH = H - KH + 1;
	static const int OW = W - KW + 1;

	LineBuffer<KH - 1, W, T> linebuf;
	WT window;
	WT filter[F];
	int16_t threshold[M];
public:
	void insert_linebuf(const int x, const T v) {
		linebuf.shift_pixels_up(x);
		linebuf.insert_bottom_row(v, x);
	}

	void slide_window(const int x, const T v) {
		T rows[KH];
#pragma HLS array_partition variable=rows

		linebuf.get_col(rows, x);
		rows[KH - 1] = v;
		insert_linebuf(x, v);

		window.shift_pixels_left();
		window.insert_right_col(rows);
	}

	WT& pack_window() {
		return window;
	}

	void load(const int weight[F * KH * KW], const int thr[M]) {
		int ptr = 0;
		for (int z = 0; z < F; z++) {
#pragma HLS pipeline
			for (int k = 0; k < KH * KW; k++) {
				filter[z].set_window(k, weight, ptr);
				ptr += C;
			}
		}

		for (int n = 0; n < M; n++) {
#pragma HLS unroll
			threshold[n] = thr[n];
		}
	}

	template <typename OT>
	void compute(fifo<WT>& ins, fifo<OT>& outs) {
		for (int xy = 0; xy < OH * OW; xy++) {
#pragma HLS pipeline
			WT val = ins.read();
			OT oval;
			for (int z = 0; z < F; z++) {
				int16_t acc = val.muladd(filter[z]);
				uint2_t m = 0;
				for (int n = 0; n < M; n++) {
					if (acc >= threshold[n]) {
						m = n + 1;
					}
				}
				bit::set<F>(oval, z, m);
			}
			outs.write(oval);
		}
	}
};

using Conv0 = Conv2D<Window_0, int2_t, 28, 28, 1, 5, 5, 16, 3>;

template <int H, int W, int KH, int KW>
void read_input(Conv0& conv, const int in[H * W], fifo<Window_0>& ins) {
	int ptr = 0;
	for (int y = 0; y < KH - 1; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			uint2_t v = in[ptr++];
			conv.insert_linebuf(x, v);
		}
	}
	for (int y = KH - 1; y < H; y++) {
#pragma HLS pipeline
		for (int x = 0; x < KW - 1; x++) {
			uint2_t v = in[ptr++];
			conv.slide_window(x, v);
		}
		for (int x = KW - 1; x < W; x++) {
			uint2_t v = in[ptr++];
			conv.slide_window(x, v);

			Window_0 oval = conv.pack_window();
			ins.write(oval);
		}
	}
}

template <int H, int W, int F>
void write_result(int out[H * W * F], fifo<int2x16_t>& outs) {
	int ptr = 0;
	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS pipeline
		int2x16_t val = outs.read();
		for (int z = 0; z < F; z++) {
			uint2_t v = bit::getu<F>(val, z);
			out[ptr++] = v;
		}
	}
}

void kernel(
	int in[HEIGHT * WIDTH],
	int weight[FILTER * KERNEL * KERNEL],
	int threshold[THRESHOLD],
	int out[OHEIGHT * OWIDTH * FILTER])
{
#pragma HLS interface axis port=in
#pragma HLS interface axis port=weight
#pragma HLS interface axis port=threshold
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=WIDTH
#pragma HLS array_partition variable=weight cyclic factor=KERNEL * KERNEL
#pragma HLS array_partition variable=out cyclic factor=FILTER

	fifo<Window_0> ins("input_fifo");
	fifo<int2x16_t> outs("output_fifo");

	Conv0 conv0;

	conv0.load(weight, threshold);

#pragma HLS dataflow
	read_input<28, 28, 5, 5>(conv0, in, ins);
	conv0.compute<int2x16_t>(ins, outs);
	write_result<24, 24, 16>(out, outs);
}
