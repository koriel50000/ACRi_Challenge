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

using bit_t = ap_uint<1>;
using int2_t = ap_int<2>;
using uint2_t = ap_uint<2>;
using int2x25_t = ap_uint<2 * KERNEL * KERNEL>;
using int1x25_t = ap_uint<1 * KERNEL * KERNEL>;
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

	template <int S>
	void setu(ap_uint<2 * S>& src, const int idx, const uint2_t& v) {
#pragma HLS inline
		int p = 2 * idx;
		src(p + 2 - 1, p) = v;
	}

	// @see HD, Figure 5-1 Counting 1-bits
	// S <= 16
	template <int S>
	int16_t population_count(int32_t x) {
		x = (x & 0x55555555) + ((x >> 1) & 0x55555555);
		x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
		x = (x & 0x0f0f0f0f) + ((x >> 4) & 0x0f0f0f0f);
		x = (x & 0x00ff00ff) + ((x >> 8) & 0x00ff00ff);
		x = (x & 0x0000ffff) + ((x >> 16) & 0x0000ffff);
		return x;
	}

	// S <= 16
	template <int S>
	int16_t multiply_add(ap_uint<S>& vu, ap_uint<S>& wp, ap_uint<S>& wn) {
		int16_t posi_count = population_count<S>(vu & wp);
		int16_t nega_count = population_count<S>(vu & wn);
		return posi_count - nega_count;
	}
} // namespace bit

class Window_0 {
private:
	static const int ROWS = 5;
	static const int COLS = 5;
public:
	int1x25_t buf;

	void shift_pixels_left() {
#pragma HLS inline
		buf >>= 1;
	}

	void insert_right_col(const bit_t value[ROWS]) {
		for (int i = 0; i < ROWS; i++) {
#pragma HLS unroll
			int idx = (i + 1) * COLS - 1;
			buf[idx] = value[i];
		}
	}
};

template <int KH, int W, typename T, typename WT>
class LineBuffer {
private:
	hls::vector<T, (KH - 1) * W> buf;
	WT window;

	void shift_pixels_up() {
#pragma HLS inline
		for (int i = 0; i < (KH - 1) * W - 1; i++) {
#pragma HLS unroll
			buf[i] = buf[i + 1];
		}
	}

	void insert_bottom_row(T value) {
#pragma HLS inline
		buf[(KH - 1) * W - 1] = value;
	}

	void get_col(T value[KH - 1]) {
#pragma HLS inline
		for (int i = 0; i < KH - 1; i++) {
#pragma HLS unroll
			value[i] = buf[i * W];
		}
	}
public:
	void insert_linebuf(const T v) {
		shift_pixels_up();
		insert_bottom_row(v);
	}

	void slide_window(const T v) {
		T rows[KH];
#pragma HLS array_partition variable=rows

		get_col(rows);
		rows[KH - 1] = v;
		shift_pixels_up();
		insert_bottom_row(v);

		window.shift_pixels_left();
		window.insert_right_col(rows);
	}

	WT& pack_window() {
		return window;
	}
};

template <typename WT, int H, int W, int C, int KH, int KW, int F, int M>
class Conv2D {
private:
	static const int OH = H - KH + 1;
	static const int OW = W - KW + 1;
public:
	template <typename OT>
	void compute(fifo<WT>& ins, fifo<OT>& outs) {
		static WT filterp[F] = {
0x01bc800, 0x0008463, 0x0002598, 0x0462300,
0x0d3b800, 0x011a000, 0x0004189, 0x1a48000,
0x1001465, 0x00a6508, 0x10a4010, 0x0006502,
0x00000ac, 0x0081095, 0x0310421, 0x1a08000,
		};
		static WT filtern[F] = {
0x000019e, 0x1ce7310, 0x0808001, 0x0018422,
0x0004050, 0x0000000, 0x1e30000, 0x0001c48,
0x0128000, 0x1000000, 0x0208002, 0x0100000,
0x0f68000, 0x1f18000, 0x00c6100, 0x000436b,
		};
		static int threshold[3] = { 1, 3, 4 };
#pragma HLS array_partition variable=filterp
#pragma HLS array_partition variable=filtern
#pragma HLS array_partition variable=threshold

		for (int xy = 0; xy < OH * OW; xy++) {
#pragma HLS pipeline
			WT val = ins.read();
			OT oval;
			for (int z = 0; z < F; z++) {
				int16_t acc = bit::multiply_add<KH * KW>(
					val, filterp[z], filtern[z]);
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

using Conv0 = Conv2D<int1x25_t, 28, 28, 1, 5, 5, 16, 3>;

template <int H, int W, int KH, int KW>
void read_input(const int in[H * W], fifo<int1x25_t>& ins) {
	LineBuffer<KH, W, bit_t, Window_0> linebuf;

	int ptr = 0;
	for (int y = 0; y < KH - 1; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			bit_t v = in[ptr++];
			linebuf.insert_linebuf(v);
		}
	}
	for (int y = KH - 1; y < H; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			bit_t v = in[ptr++];
			linebuf.slide_window(v);

			if (x >= KW - 1) {
				Window_0 oval = linebuf.pack_window();
				ins.write(oval.buf);
			}
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
#pragma HLS unroll
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
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=WIDTH
#pragma HLS array_partition variable=out cyclic factor=FILTER

	fifo<int1x25_t> ins("input_fifo");
	fifo<int2x16_t> outs("output_fifo");

	Conv0 conv0;

#pragma HLS dataflow
	read_input<28, 28, 5, 5>(in, ins);
	conv0.compute<int2x16_t>(ins, outs);
	write_result<24, 24, 16>(out, outs);
}
