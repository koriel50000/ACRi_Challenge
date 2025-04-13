/*
 * 4bit量子化および演算回路再利用の検証
 * ・weightを1bit符号＋3bit指数部の4bitで表現(0,0.125,0.25,0.5,1,2,4,8,NA,-0.125,-0.25,-0.5,-1,-2,-4,-8)
 * ・バッチ正規化後のactivationを1bit符号＋3bit仮数部の4bitで表現(0,1,2,3,4,5,6,7,NA,-1,-2,-3,-4,-5,-6,-7)
 * ・乗算は符号なし3bitの掛け算を、6入力LUTが6個のテーブル参照で計算
 * ・演算回路は最大サイズのConv,Maxpoolを用意し、引数で行列サイズを指定して再利用
 * ・ダブルバッファリングで演算結果を一時保存
 */
#include "kernel.hpp"
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_streamofblocks.h>
#include <hls_vector.h>

#define I4(i) int_t<16>(i)

const int WIDTH = 32;
const int HEIGHT = 32;
const int CHANNEL = 16;
const int FILTER = 16;

const int KERNEL = 5;
const int THRESHOLD = 3;

using uint4_t = ap_uint<4>;
using uint6_t = ap_uint<6>;
template <typename T>
using fifo = hls::stream<T>;

template <int N, int W = 4>
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

uint6_t mul66(const uint6_t i) {
	static const uint6_t table[] = {
		0,	0,	0,	0,	0,	0,	0,	0,
		0,	0,	0,	1,	1,	2,	4,	8,
		0,	0,	1,	1,	2,	4,	8,	16,
		0,	0,	1,	2,	3,	6,	12,	24,
		0,	1,	1,	2,	4,	8,	16,	32,
		0,	1,	1,	3,	5,	10,	20,	40,
		0,	1,	2,	3,	6,	12,	24,	48,
		0,	1,	2,	4,	7,	14,	28,	56,
	};
	return table[i];
}

int8_t mul(const uint4_t v, const uint4_t w) {
	uint6_t oval = mul66((v(2, 0), w(2, 0)));
	return (v[3] ^ w[3]) == 1 ? (-oval).to_int() : oval.to_int();
}

template <int N>
int16_t muladd(const int n, const int_t<N> vu, const int_t<N> wi) {
	static int16_t t[N];
#pragma HLS array_partition variable=t

	for (int i = 0; i < N; i++) {
		// @see UG1399, Vitis HLS Coding Styles > Loops > Variable Loop Bounds
#pragma HLS unroll
		if (i >= n) break;
		t[i] = mul(vu[i], wi[i]);
	}

	for (int d = 1; d < N; d *= 2) {
		if (d >= n) break;
		for (int i = 0; i < N; i += d * 2) {
#pragma HLS unroll
			if (i >= n) break;
			t[i] += t[i + d];
		}
	}
	return t[0];
}

uint4_t batch_norm(const int16_t acc, const int thr[], bool relu) {
// 	static const uint4_t indexTable[] = {
// 		0, 1, 2, 4, 7, 3, 6, 5,
// 	};
// #pragma HLS array_partition variable=indexTable
// 	// @see HD, Figure 5-26. Number of trailing zeros using a de Brujin cycle.
// 	// https://en.wikipedia.org/wiki/De_Bruijn_sequence
	
	ap_uint<1> b0 = acc >= thr[0];
	ap_uint<1> b1 = acc >= thr[1];
	ap_uint<1> b2 = acc >= thr[2];
	ap_uint<1> b3 = acc >= thr[3];
	ap_uint<1> b4 = acc >= thr[4];
	ap_uint<1> b5 = acc >= thr[5];
	ap_uint<1> b6 = acc >= thr[6];
	ap_uint<8> bits = (0, b6, b5, b4, b3, b2, b1, b0);
	// return indexTable[((bits + 1) * 0x17)(7, 5)];
	// @see UG1399, Vitis HLS Coding Styles > Functions > C/C++ Builtin Functions
	return __builtin_ctz(bits + 1);
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

template <int W, int KN, typename T, typename WT>
class LineBuffer {
private:
	T buf_[W * (KN - 1)];
	Window<KN, KN, T, WT> window_;
	int width_;

	void shift_pixels_up() {
#pragma HLS inline
		for (int i = 0; i < W * (KN - 1) - 1; i++) {
#pragma HLS unroll
			buf_[i] = buf_[i + 1];
		}
	}

	void insert_bottom_row(T value) {
#pragma HLS inline
		buf_[width_ * (KN - 1) - 1] = value;
	}

	void get_col(T value[KN - 1]) {
#pragma HLS inline
		for (int i = 0; i < KN - 1; i++) {
#pragma HLS unroll
			value[i] = buf_[i * width_];
		}
	}
public:
	LineBuffer(int w = W) : width_(w) {}

	void insert_linebuf(const T v) {
		shift_pixels_up();
		insert_bottom_row(v);
	}

	void slide_window(const T v) {
		T rows[KN];
#pragma HLS array_partition variable=rows

		get_col(rows);
		rows[KN - 1] = v;
		shift_pixels_up();
		insert_bottom_row(v);

		window_.shift_pixels_left();
		window_.insert_right_col(rows);
	}

	WT& get_window() {
		return window_.get_buf();
	}
};

template <int H, int W, int C, int F, int KN, int PD = 0, int ST = 1>
class Conv2D {
private:
	using T = int_t<C>;
	using WT = hls::vector<T, KN * KN>;

	void windowize(const int h, const int w, const T inb[], fifo<WT>& pips) {
		LineBuffer<W + PD, KN, T, WT> linebuf(w);

		int x = 0 - (KN - 1);
		int y = 0 - (KN - 1);
		for (int i = 0; i < (W + PD) * (H + PD * 2) + PD; i++) {
#pragma HLS pipeline
			if (i >= (w + PD) * (h + PD * 2) + PD) break;
			T val;
			if (0 - (KN - 1) + PD <= x && x < w - (KN - 1) + PD
				&& 0 - (KN - 1) + PD <= y && y < h - (KN - 1) + PD)
			{
				val = inb[(y + (KN - 1)) * WIDTH + (x + (KN - 1))];
			}
			else {
				val = 0;
			}
			if (i < (w + PD) * (KN - 1) - PD) {
				linebuf.insert_linebuf(val);
			}
			else {
				linebuf.slide_window(val);
			}
			if (0 <= x && 0 <= y && x % ST == 0 && y % ST == 0) {
				WT oval = linebuf.get_window();
				pips.write(oval);
			}
			x++;
			if (x >= w - (KN - 1) + PD * 2) {
				x = 0 - (KN - 1) + PD;
				y++;
			}
		}
	}

	void conv(const int h, const int w, const int c, const int f, const T wi[], const int thr[],
		T outb[], fifo<WT>& pips)
	{
		for (int y = 0; y < H - (KN - 1); y++) {
			if (y >= h - (KN - 1)) break;
			for (int x = 0; x < W - (KN - 1); x++) {
				if (x >= w - (KN - 1)) break;
				WT val = pips.read();
				T oval;
				for (int j = 0; j < F; j++) {
#pragma HLS pipeline
					if (j >= f) break;
					int16_t acc = 0;
					for (int k = 0; k < KN * KN; k++) {
						acc += muladd<C>(c, val[k], wi[j * KN * KN + k]);
					}
					oval[j] = batch_norm(acc, thr, true);
				}
				outb[y * WIDTH + x] = oval;
			}
		}
	}
public:
	void compute(const int h, const int w, const int c, const int f, const T wi[], const int thr[],
		const T inb[], T outb[])
	{
		fifo<WT> pips("pipe_fifo");

#pragma HLS dataflow
		windowize(h, w, inb, pips);
		conv(h, w, c, f, wi, thr, outb, pips);
	}
};

template <int H, int W, int C>
class MaxPool2x2 {
private:
	using T = int_t<C>;

	void maxpool(const int c, const T v1, const T v2, T& ov) {
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			if (z >= c) break;
			ov[z] = (v1[z] > v2[z]) ? v1[z] : v2[z];
		}
	}

	void compute_h(const int h, const int w, const int c, const T inb[], fifo<T>& pips) {
		for (int y = 0; y < H; y++) {
			if (y >= h) break;
			for (int x = 0; x < W; x += 2) {
#pragma HLS pipeline
				if (x >= w) break;
				T val1 = inb[y * WIDTH + x];
				T val2 = inb[y * WIDTH + x + 1];
				T oval;
				maxpool(c, val1, val2, oval);
				pips.write(oval);
			}
		}
	}

	void compute_v(const int oh, const int ow, const int c, T outb[], fifo<T>& pips) {
		static T buf[W / 2];
#pragma HLS array_partition variable=buf

		for (int y = 0; y < H; y++) {
			if (y >= oh) break;
			for (int x = 0; x < W; x++) {
#pragma HLS pipeline
				if (x >= ow) break;
				buf[x] = pips.read();
			}
			for (int x = 0; x < W; x++) {
#pragma HLS pipeline
				if (x >= ow) break;
				T val1 = buf[x];
				T val2 = pips.read();
				T oval;
				maxpool(c, val1, val2, oval);
				outb[y * WIDTH + x] = oval;
			}
		}
	}

public:
	void compute(const int h, const int w, const int c, const T inb[], T outb[]) {
		fifo<T> pips("pipe_fifo");

#pragma HLS dataflow
		compute_h(h, w, c, inb, pips);
		compute_v(h / 2, w / 2, c, outb, pips);
	}
};

template <int H, int W, int C, typename T>
void read_input(const int in[H * W * C], T inb[]) {
	int ptr = 0;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
			T val;
			for (int z = 0; z < C; z++) {
#pragma HLS unroll
				val[z] = in[ptr++];
			}
			inb[y * WIDTH + x] = val;
		}
	}
}

template <int H, int W, int C, typename T>
void write_result(int out[H * W * C], const T outb[]) {
	int ptr = 0;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
			T val = outb[y * WIDTH + x];
			for (int z = 0; z < C; z++) {
#pragma HLS unroll
				out[ptr++] = val[z];
			}
		}
	}
}

void kernel(
	int in[12 * 12 * 16],
	int weight[16 * 5 * 5 * 16],
	int threshold[3],
	int out[4 * 4 * 16])
{
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=16
#pragma HLS array_partition variable=out cyclic factor=16

	static int_t<CHANNEL> even_buf[HEIGHT * WIDTH];
	static int_t<CHANNEL> odd_buf[HEIGHT * WIDTH];
#pragma HLS array_partition variable=even_buf cyclic factor=WIDTH
#pragma HLS array_partition variable=odd_buf cyclic factor=WIDTH

	static int_t<CHANNEL> conv_wi[FILTER * KERNEL * KERNEL] = {
I4(0x4000000000000000), I4(0x00000000000000c0), I4(0x04000000000000c0), I4(0x040c0000000000cc), I4(0x440c00000c000000),
I4(0x0004000000000000), I4(0x0c040000000000c0), I4(0x000c000000000000), I4(0x040c000000000000), I4(0x040c0000000000c0),
I4(0xcc040000000c0000), I4(0xcc00000000000000), I4(0x000c000000404000), I4(0xc00c004000000040), I4(0xc0000c00c0400000),
I4(0xcccc00004c000000), I4(0xc0cc00000c004000), I4(0xc00000000c044040), I4(0xc00c0000c0000040), I4(0xc00000000000000c),
I4(0xc00c04000c004000), I4(0xc000004000000440), I4(0xc00040000000004c), I4(0x00000000c0000000), I4(0x00000000000000c4),
I4(0x0c0400000000000c), I4(0x0c04000000cc0040), I4(0x0000000000000000), I4(0x0000000000c0000c), I4(0x0000000000000000),
I4(0x4c040c0000000000), I4(0x4c0400c00000c000), I4(0x0000c00000c000c0), I4(0x4c0c000000000000), I4(0x000c000000000000),
I4(0x400000c040000c04), I4(0x40c400000c0000c0), I4(0x000004004c000000), I4(0x400c0000400000c0), I4(0x0000000000400000),
I4(0x40000040004000c4), I4(0x0000000040004000), I4(0xc00c000000044404), I4(0xc00c000000000400), I4(0x0000000400000000),
I4(0x00000000040000c4), I4(0xc0040000c0000440), I4(0xc4404000c0000040), I4(0x00404400c0000000), I4(0x000c00000004c0c0),
I4(0x0c00000c00400000), I4(0x0c040000000000c0), I4(0x00040000c000c0c0), I4(0xc00000c0000c0000), I4(0x000c0c0000000000),
I4(0x040c0000004400c0), I4(0x4000c000000000c0), I4(0x000400c0400000c0), I4(0x00000000400000c0), I4(0x0000c00c00000000),
I4(0x00000004004000c0), I4(0x40000000004000c0), I4(0x40000000400000c0), I4(0x4000000040000004), I4(0x4c00000000000004),
I4(0xcc444004c40000c0), I4(0x004000040400c0c0), I4(0x00440000000000c4), I4(0x4004000000000004), I4(0x4c04000000000004),
I4(0x0c000000c40000c0), I4(0x00000c00c4c0c0c0), I4(0x00000000c400c000), I4(0x0c040000c40cc000), I4(0x0c040000040cc0c0),
I4(0x0400000400000040), I4(0x0000000040000040), I4(0x0c00000c00000000), I4(0x000000000c0000c0), I4(0x40cc00000c0000c0),
I4(0x4400000040000040), I4(0x0000000000000000), I4(0x0000000000000000), I4(0x00000000000000c0), I4(0x040c000000000000),
I4(0x0400000000000040), I4(0xc0000000000cc000), I4(0x0000000000000000), I4(0xc00c000000004000), I4(0x000c00000c000000),
I4(0x000000c00000c000), I4(0xc0c000004c000c00), I4(0xc00c00004c004000), I4(0xc00c004000404040), I4(0xc00c0000c0000040),
I4(0xc00c00000c000cc0), I4(0xc00c000c0c0040c4), I4(0xc000000000404440), I4(0xc00000040000004c), I4(0xc0000004c0000040),
I4(0x0000000c0c0cc0c0), I4(0x0c00000000000000), I4(0xc000400000004400), I4(0x0000004400000400), I4(0x0000000000040000),
I4(0x000cc00cc00cc000), I4(0x0c00000040c0000c), I4(0xc000040000004000), I4(0xc000040000000000), I4(0xc00c000000000000),
I4(0x00c000000c0c0c40), I4(0x0000000c0cc00000), I4(0x000004000c000040), I4(0xc000004000444440), I4(0xc0000040c0000000),
I4(0x00c00000000ccc40), I4(0x0c00000c0c000000), I4(0x000c04400c004040), I4(0xc4000040c040004c), I4(0xc0040000c000000c),
I4(0x0000c00000000000), I4(0x0000000000000040), I4(0x04000400c0000040), I4(0x0000000000000000), I4(0x0000000040000000),
I4(0x0004000400000000), I4(0x00000000000000c0), I4(0x040c00000c000000), I4(0x000c000c0c000040), I4(0x000c00000c000c00),
I4(0x000400000000c0c0), I4(0x0000000000000000), I4(0x000c00004c000040), I4(0x040c00000c404040), I4(0x0000000000000040),
I4(0x0c04000000000000), I4(0xc00c00000c000040), I4(0xc40c000000004040), I4(0xc40c00000c004040), I4(0xc4cc0c0000000000),
I4(0xc0cc000000000000), I4(0xc00c000000004040), I4(0xc40c00000000004c), I4(0xc00c0004c0044000), I4(0x04cc00000c000c00),
I4(0xc00000000c000040), I4(0xc000000000000040), I4(0xc40c00000000044c), I4(0x000c0040000000c0), I4(0x0400000000040000),
I4(0xc0cc00cc4c000000), I4(0x00cc000c0c004000), I4(0x00cc000040000044), I4(0x4000004000004000), I4(0x4044004404000000),
I4(0xc00c00000cc00000), I4(0xc00c400000004400), I4(0x0000000000400040), I4(0x0000000004000000), I4(0x0c440c00000c0400),
I4(0xc00c000c00c0000c), I4(0x004c000000000000), I4(0x44400000c4000000), I4(0x4004000004000000), I4(0x0004ccc000cc00cc),
I4(0x040c0000cc00000c), I4(0x00000000c0000000), I4(0x4400000000000000), I4(0x40040000400000c0), I4(0x4c00cc0c0c000cc0),
I4(0x0000040c00000000), I4(0x4000000040004000), I4(0x4000400000440004), I4(0x4000000000400004), I4(0x0c0000000000c000),
I4(0x0400000000000000), I4(0x000000000000000c), I4(0x0000000000c00040), I4(0x00000000000c0000), I4(0x000c0000c0000000),
I4(0x0000000000000040), I4(0x000000c00000004c), I4(0xc00c000c0cc00c4c), I4(0x000c00000c000040), I4(0x000c00000c000040),
I4(0x0000000000000040), I4(0x0000000000000040), I4(0x040c04004c004040), I4(0x040c00000c000040), I4(0x000000000000004c),
I4(0x00ccc0000c0c0c40), I4(0x000c00000c004040), I4(0xc00c00000c004040), I4(0x000c000040404040), I4(0x040c00000c000000),
I4(0x00c0c00000000040), I4(0xc000040000004000), I4(0x040c000000000044), I4(0x000c444000000040), I4(0x440c000000044000),
I4(0x00cc000c0c0c4c00), I4(0x000c000c0c000040), I4(0xc00c000000000040), I4(0x0000000000000000), I4(0x00000cc0c000c00c),
I4(0x00cc00004c000040), I4(0x0400000040000040), I4(0x0000000000000000), I4(0x0000000000000000), I4(0x4000ccc00000cc0c),
I4(0x000c000000000000), I4(0x000c00000c040440), I4(0xc40c000000440004), I4(0x400c000000400000), I4(0x4c00000040040c04),
I4(0xc0000000000000c0), I4(0x000c0000000000c0), I4(0x0000000000400000), I4(0x0004000040000004), I4(0x4c04004004000404),
I4(0xc00000000000000c), I4(0x0c000000c00000c0), I4(0x00400000c000c0c0), I4(0x40440000c400c400), I4(0x40440c00c400c000),
I4(0x0000000000c000c0), I4(0x440c00000c000004), I4(0x000c400c00400000), I4(0x0c00404004000000), I4(0x0c04404400000400),
I4(0x0000040000c000c0), I4(0x0000400000000040), I4(0x00400000c0000040), I4(0x00000004c0000000), I4(0x0c000c04c00c000c),
I4(0xc00c0000000040cc), I4(0x0000000000400040), I4(0x000c0c00c0000000), I4(0x0000ccc0c00cc000), I4(0x0000c0cc00c0c000),
I4(0x000c004000000000), I4(0xc0000004c0000000), I4(0xc0000000c4000040), I4(0x000c0000c0cc000c), I4(0x0000c00c0cc00000),
I4(0xc400000000004040), I4(0x0000000000000040), I4(0x040000004000c000), I4(0x40c0000c40000000), I4(0x400c04004c004004),
I4(0x0c0c000c000000c0), I4(0x0000000004000000), I4(0x4044004004000000), I4(0x4004400044000404), I4(0x40040044040004c0),
I4(0x000000000400c0c0), I4(0x40040c00c40cc000), I4(0x4c040c0004ccc0c0), I4(0x0c0400000400c0cc), I4(0x0c040cc000cc000c),
I4(0x4004c000040000c0), I4(0x4004cc000400c0c4), I4(0x4004c00c440000c4), I4(0x4c04c00000000cc0), I4(0x4000c0c00c000c00),
I4(0x4c04000000000cc0), I4(0x4000000000000c00), I4(0x400000000c000c04), I4(0x00000000000000c4), I4(0x0000000000000000),
I4(0x4000000040400000), I4(0x0400000040400040), I4(0x0000000000404040), I4(0x00000000c0000000), I4(0x000000c0c000000c),
I4(0xcc00000000c0004c), I4(0xc0000000c000004c), I4(0xc0000cc0c00c0040), I4(0xc00c000c00c00040), I4(0xc00c000c0c00400c),
I4(0x4000c00c00000000), I4(0x0004c0c000000040), I4(0x0000c0c0000c0c40), I4(0x0c0c00c00c000000), I4(0x000c000c0c00000c),
I4(0x0c00000000400004), I4(0x400400004000c000), I4(0x0cc000000c000000), I4(0x00c000004c0000c4), I4(0x0000400000404000),
I4(0x0000000400000000), I4(0x00000000400000c0), I4(0x0000000040004004), I4(0x0000040000004440), I4(0xc000000000400040),
I4(0x0c000004000000c0), I4(0x0000000000000000), I4(0x0000040000004040), I4(0x0000440000040040), I4(0x000c0000c0000040),
I4(0x0c04400000c004c0), I4(0x0c040000c4c000c0), I4(0x0000000000000000), I4(0x0400000000000000), I4(0x04cc00000c000000),
I4(0x0c0400c0000000c0), I4(0x0c04000000000000), I4(0x000000004c000000), I4(0x000c00400c400040), I4(0x000c000000000040),
I4(0x0c00c00c4c0c0040), I4(0x000c00004c000040), I4(0xc40c00000c000040), I4(0xc00c40000c004040), I4(0xc40c000000000040),
I4(0x0c00000c00004040), I4(0xc00c000000000440), I4(0xc40c00000c004040), I4(0xc40c0400c000404c), I4(0xc40c00000c000000),
I4(0xcc000000000c0000), I4(0xc4000000c000004c), I4(0xc40c00000000004c), I4(0x000c044000000000), I4(0x00000440000000c4),
I4(0x00040004000000c0), I4(0x000000000000c000), I4(0x000c0000c0000000), I4(0xc40c000000000c00), I4(0x000cc00c0c000c00),
I4(0x0c00000000000000), I4(0x0000000000000000), I4(0x000c00000c000000), I4(0x000c00000c000000), I4(0x04c0000000000000),
I4(0xc000000000000040), I4(0xc00c000000000040), I4(0xc00c00000c000040), I4(0xc00c000000000040), I4(0xc40c000000000000),
I4(0x00c000000c400000), I4(0xc000000000004000), I4(0xc000000000004040), I4(0xc00c00000c000000), I4(0x00cc000040000000),
I4(0x000000000c400004), I4(0xc000004000040000), I4(0xc040000000044000), I4(0x00000400000400c0), I4(0x40004000004400c4),
I4(0x4cc00000404400c0), I4(0x40000000440000c0), I4(0x00000000400000c0), I4(0x000000cc000000c4), I4(0x00c4cc0c000000c0),
I4(0x4c000000000000c4), I4(0x40440044040000c4), I4(0x00000000040400c4), I4(0x04000000000000c4), I4(0x00000000000000c0),
I4(0x0c440000040004c0), I4(0x0c040000040004c0), I4(0x0c440004040000c0), I4(0x00040000040000c0), I4(0x0000000000000004),
I4(0x00000c0004c000c0), I4(0x000400000000c0cc), I4(0x00040000040000c0), I4(0x0004000004c00000), I4(0x0044400000000000),
I4(0x4c000000000000c0), I4(0x4c000c00000000c0), I4(0x4c040000000000c0), I4(0x0004000c00cc000c), I4(0xc004000004c00000),
I4(0x000c00000c000000), I4(0x40c0000c40000040), I4(0x0c00000000000000), I4(0x0c00000000000000), I4(0xc0000000c0000c00),
I4(0x0000000040040004), I4(0x04cc00004c400c00), I4(0x4000000000000000), I4(0x00000c0000000000), I4(0x00c00cc0c00000cc),
I4(0x00400000000000c0), I4(0x00000000404400c0), I4(0x040c00004c040004), I4(0x40cc00000c000000), I4(0x04c0c0c00c00cc00),
I4(0xc0044000000000c0), I4(0x00400000040000c0), I4(0x000c0000000400c4), I4(0x00000000000400c4), I4(0x400000004c040c04),
I4(0x0000000000cc00cc), I4(0x00000000000000cc), I4(0x00044004c00000c0), I4(0x0c400000040000c0), I4(0x00004000004004c4),
	};
	static int conv_thr[7] = { 3, 9, 14, 0x7fff, 0x7fff, 0x7fff, 0x7fff };
#pragma HLS array_partition variable=conv_wi cyclic factor=KERNEL*KERNEL
#pragma HLS array_partition variable=conv_thr

	Conv2D<HEIGHT,WIDTH,CHANNEL,FILTER,KERNEL> conv;
	MaxPool2x2<HEIGHT,WIDTH,CHANNEL> maxpool;

	read_input<12,12,16,int_t<CHANNEL>>(in, even_buf);
	conv.compute(12, 12, 16, 16, conv_wi, conv_thr, even_buf, odd_buf);
	maxpool.compute(8, 8, 16, odd_buf, even_buf);
	write_result<4,4,16,int_t<CHANNEL>>(out, even_buf);
}
