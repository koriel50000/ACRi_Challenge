#include "kernel.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

const int WIDTH = 28;
const int HEIGHT = 28;

const int FILTER = 16;
const int KERNEL = 5;
const int THRESHOLD = 3;

const int OWIDTH = WIDTH - KERNEL + 1;
const int OHEIGHT = HEIGHT - KERNEL + 1;

const ap_uint<1> b0w1 = 0;

using bit_t = ap_uint<1>;
using int2_t = ap_int<2>;
using uint2_t = ap_uint<2>;
using int3_t = ap_int<3>;
using uint3_t = ap_uint<3>;
using int4_t = ap_int<4>;
using uint6_t = ap_uint<6>;

template <int W, int N>
class int_t {
private:
	ap_uint<W*N> buf_;
public:
	int_t() : buf_(0) {}
	int_t(int i) : buf_(i) {}
	int_t(unsigned int ui) : buf_(ui) {}
	int_t(long l) : buf_(l) {}
	int_t(unsigned long ul) : buf_(ul) {}
	int_t(const char* s) : buf_(s) {}

	inline ap_range_ref<W*N, false> operator[](size_t index) const {
		assert(index < N);
		return buf_(W * index + W - 1, W * index);
	}

	inline ap_range_ref<W*N, false> operator[](size_t index) {
		assert(index < N);
		return buf_(W * index + W - 1, W * index);
	}
};

template <typename T>
using fifo = hls::stream<T>;

void mac63(uint6_t i, int3_t& o) {
	static const int3_t table[] = {
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, -1, 0, -1, 1, 0, 1, 0,
		0, -1, 0, -1, 1, 0, 1, 0,
		0, 0, -1, -1, 0, 0, -1, -1,
		1, 1, 0, 0, 1, 1, 0, 0,
		0, -1, -1, -2, 1, 0, 0, -1,
		1, 0, 0, -1, 2, 1, 1, 0,
	};
	o = table[i];
}

void ac64(uint6_t i, int4_t& o) {
	static const int4_t table[] = {
		0, 1, 2, 3, -4, -3, -2, -1,
		1, 2, 3, 4, -3, -2, -1, 0,
		2, 3, 4, 5, -2, -1, 0, 1,
		3, 4, 5, 6, -1, 0, 1, 2,
		-4, -3, -2, -1, -8, -7, -6, -5,
		-3, -2, -1, 0, -7, -6, -5, -4,
		-2, -1, 0, 1, -6, -5, -4, -3,
		-1, 0, 1, 2, -5, -4, -3, -2,
	};
	o = table[i];
}

int16_t muladd25(int_t<1,25> vu, int_t<1,25> wp, int_t<1,25> wn) {
	int3_t c0100, c0302, c0504, c0706, c0908, c1110, c1312, c1514;
	int3_t c1716, c1918, c2120, c2322, c24;

	mac63((vu[ 1], vu[ 0], wp[ 1], wp[ 0], wn[ 1], wn[ 0]), c0100);
	mac63((vu[ 3], vu[ 2], wp[ 3], wp[ 2], wn[ 3], wn[ 2]), c0302);
	mac63((vu[ 5], vu[ 4], wp[ 5], wp[ 4], wn[ 5], wn[ 4]), c0504);
	mac63((vu[ 7], vu[ 6], wp[ 7], wp[ 6], wn[ 7], wn[ 6]), c0706);
	mac63((vu[ 9], vu[ 8], wp[ 9], wp[ 8], wn[ 9], wn[ 8]), c0908);
	mac63((vu[11], vu[10], wp[11], wp[10], wn[11], wn[10]), c1110);
	mac63((vu[13], vu[12], wp[13], wp[12], wn[13], wn[12]), c1312);
	mac63((vu[15], vu[14], wp[15], wp[14], wn[15], wn[14]), c1514);
	mac63((vu[17], vu[16], wp[17], wp[16], wn[17], wn[16]), c1716);
	mac63((vu[19], vu[18], wp[19], wp[18], wn[19], wn[18]), c1918);
	mac63((vu[21], vu[20], wp[21], wp[20], wn[21], wn[20]), c2120);
	mac63((vu[23], vu[22], wp[23], wp[22], wn[23], wn[22]), c2322);
	mac63((b0w1, vu[24], b0w1, wp[24], b0w1, wn[24]), c24);

	int4_t c0, c1, c2, c3, c4, c5;

	ac64((c0100, c0302), c0);
	ac64((c0504, c0706), c1);
	ac64((c0908, c1110), c2);
	ac64((c1312, c1514), c3);
	ac64((c1716, c1918), c4);
	ac64((c2120, c2322), c5);

	return ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + c24);
}

template <int ROWS, int COLS, typename T, typename WT>
class Window {
private:
	WT buf_;
public:
	void shift_pixels_left() {
#pragma HLS inline
		for (int i = 0; i < ROWS * COLS - 1; i++) {
#pragma HLS unroll
			buf_[i] = buf_[i + 1];
		}
	}

	void insert_right_col(const T value[ROWS]) {
#pragma HLS inline
		for (int i = 0; i < ROWS; i++) {
#pragma HLS unroll
			int idx = (i + 1) * COLS - 1;
			buf_[idx] = value[i];
		}
	}

	WT& get_buf() {
		return buf_;
	}
};

template <int KH, int W, typename T, typename WT>
class LineBuffer {
private:
	hls::vector<T, (KH - 1) * W> buf_;
	Window<KH, KH, T, WT> window_;

	void shift_pixels_up() {
#pragma HLS inline
		for (int i = 0; i < (KH - 1) * W - 1; i++) {
#pragma HLS unroll
			buf_[i] = buf_[i + 1];
		}
	}

	void insert_bottom_row(T value) {
#pragma HLS inline
		buf_[(KH - 1) * W - 1] = value;
	}

	void get_col(T value[KH - 1]) {
#pragma HLS inline
		for (int i = 0; i < KH - 1; i++) {
#pragma HLS unroll
			value[i] = buf_[i * W];
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

		window_.shift_pixels_left();
		window_.insert_right_col(rows);
	}

	WT& get_window() {
		return window_.get_buf();
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
		static WT fp[F] = {
0x01bc800, 0x0008463, 0x0002598, 0x0462300,
0x0d3b800, 0x011a000, 0x0004189, 0x1a48000,
0x1001465, 0x00a6508, 0x10a4010, 0x0006502,
0x00000ac, 0x0081095, 0x0310421, 0x1a08000,
		};
		static WT fn[F] = {
0x000019e, 0x1ce7310, 0x0808001, 0x0018422,
0x0004050, 0x0000000, 0x1e30000, 0x0001c48,
0x0128000, 0x1000000, 0x0208002, 0x0100000,
0x0f68000, 0x1f18000, 0x00c6100, 0x000436b,
		};
		static int thr[] = { 1, 3, 4 };
#pragma HLS array_partition variable=fp
#pragma HLS array_partition variable=fn
#pragma HLS array_partition variable=thr

		for (int xy = 0; xy < OH * OW; xy++) {
#pragma HLS pipeline
			WT val = ins.read();
			OT oval;
			for (int z = 0; z < F; z++) {
				int16_t acc = muladd25(val, fp[z], fn[z]);
				uint2_t m = 0;
				for (int n = 0; n < M; n++) {
					if (acc >= thr[n]) {
						m = n + 1;
					}
				}
				oval[z] = m;
			}
			outs.write(oval);
		}
	}
};

using Conv0 = Conv2D<int_t<1,25>, 28, 28, 1, 5, 5, 16, 3>;

template <int H, int W, int KH, int KW>
void read_input(const int in[H * W], fifo<int_t<1,25>>& ins) {
	LineBuffer<KH, W, bit_t, int_t<1,25>> linebuf;

	for (int y = 0; y < KH - 1; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			bit_t v = in[y * W + x];
			linebuf.insert_linebuf(v);
		}
	}
	for (int y = KH - 1; y < H; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			bit_t v = in[y * W + x];
			linebuf.slide_window(v);

			if (x >= KW - 1) {
				int_t<1,25> oval = linebuf.get_window();
				ins.write(oval);
			}
		}
	}
}

template <int H, int W, int F>
void write_result(int out[H * W * F], fifo<int_t<2,16>>& outs) {
	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS pipeline
		int_t<2,16> val = outs.read();
		for (int z = 0; z < F; z++) {
#pragma HLS unroll
			out[xy * F + z] = val[z];
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

	fifo<int_t<1,25>> ins("input_fifo");
	fifo<int_t<2,16>> outs("output_fifo");

	Conv0 conv0;

#pragma HLS dataflow
	read_input<28, 28, 5, 5>(in, ins);
	conv0.compute<int_t<2,16>>(ins, outs);
	write_result<24, 24, 16>(out, outs);
}
