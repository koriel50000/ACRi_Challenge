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
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>

const int WIDTH = 16;
const int HEIGHT = 16;
const int CHANNEL = 16;
const int FILTER = 16;

const int KERNEL = 5;
const int THRESHOLD = 3;

using uint4_t = ap_uint<4>;
using uint6_t = ap_uint<6>;
template <typename T>
using win_t = hls::vector<T, KERNEL * KERNEL>;

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
int16_t muladd(const int n, const int_t<4,N> vu, const int_t<4,N> wi) {
	static int16_t t[N];
#pragma HLS array_partition variable=t

	for (int i = 0; i < N; i++) {
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
	static const uint4_t indexTable[] = {
		0, 1, 2, 4, 7, 3, 6, 5,
	};
#pragma HLS array_partition variable=indexTable
	// @see HD, Figure 5-26. Number of trailing zeros using a de Brujin cycle.
	// https://en.wikipedia.org/wiki/De_Bruijn_sequence
	
	ap_uint<1> b0 = acc >= thr[0];
	ap_uint<1> b1 = acc >= thr[1];
	ap_uint<1> b2 = acc >= thr[2];
	ap_uint<1> b3 = acc >= thr[3];
	ap_uint<1> b4 = acc >= thr[4];
	ap_uint<1> b5 = acc >= thr[5];
	ap_uint<1> b6 = acc >= thr[6];
	ap_uint<8> bits = (0, b6, b5, b4, b3, b2, b1, b0);
	return indexTable[((bits + 1) * 0x17)(7, 5)];
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

template <int H, int W, int C, int F, int KN, typename T, typename WT, int PD = 0, int ST = 1>
class Conv2D {
private:
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
	void read(const int c, const int f, const int weight[], const int threshold[], T wi[], int thr[]) {
		int ptr = 0;
		for (int j = 0; j < F; j++) {
#pragma HLS pipeline
			if (j >= f) break;
			for (int k = 0; k < KN * KN; k++) {
				T val;
				for (int i = 0; i < C; i++) {
#pragma HLS unroll
					if (i >= c) break;
					val[i] = (weight[ptr++] << 2) & 0xf;
				}
				wi[j * KN * KN + k] = val;
			}
		}

		for (int i = 0; i < THRESHOLD; i++) {
#pragma HLS unroll
			thr[i] = threshold[i];
		}
	}

	void compute(const int h, const int w, const int c, const int f, const T wi[], const int thr[],
		const T inb[], T outb[])
	{
		fifo<WT> pips("pipe_fifo");

#pragma HLS dataflow
		windowize(h, w, inb, pips);
		conv(h, w, c, f, wi, thr, outb, pips);
	}
};

template <int H, int W, int C, typename T>
class MaxPool2x2 {
private:
	void maxpool(const int c, const T v1, const T v2, T& ov) {
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			if (z >= c) break;
			ov[z] = (v1[z] > v2[z]) ? v1[z] : v2[z];
		}
	}

	void compute_h(const int h, const int w, const int c, const T inb[], fifo<T>& pips) {
		for (int y = 0; y < H; y++) {
#pragma HLS pipeline
			if (y >= h) break;
			for (int x = 0; x < W; x += 2) {
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
#pragma HLS pipeline
			if (y >= oh) break;
			for (int x = 0; x < W; x++) {
				if (x >= ow) break;
				buf[x] = pips.read();
			}
			for (int x = 0; x < W; x++) {
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

template <int H, int W, int C>
void read_input(const int in[H * W * C], int_t<4,CHANNEL> inb[]) {
	int ptr = 0;
	for (int y = 0; y < H; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			int_t<4,CHANNEL> val;
			for (int z = 0; z < C; z++) {
#pragma HLS unroll
				val[z] = in[ptr++];
			}
			inb[y * WIDTH + x] = val;
		}
	}
}

template <int H, int W, int C>
void write_result(int out[H * W * C], const int_t<4,CHANNEL> outb[]) {
	int ptr = 0;
	for (int y = 0; y < H; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			int_t<4,CHANNEL> val = outb[y * WIDTH + x];
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
#pragma HLS interface axis port=weight
#pragma HLS interface axis port=threshold
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=16
#pragma HLS array_partition variable=weight cyclic factor=16
#pragma HLS array_partition variable=threshold cyclic factor=3
#pragma HLS array_partition variable=out cyclic factor=16

	static int_t<4,CHANNEL> even_buf[HEIGHT * WIDTH];
	static int_t<4,CHANNEL> odd_buf[HEIGHT * WIDTH];
#pragma HLS array_partition variable=even_buf cyclic factor=WIDTH
#pragma HLS array_partition variable=odd_buf cyclic factor=WIDTH

	static int_t<4,CHANNEL> conv_wi[FILTER * KERNEL * KERNEL];
	static int conv_thr[7] = { 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff };
#pragma HLS array_partition variable=conv_wi cyclic factor=KERNEL*KERNEL
#pragma HLS array_partition variable=conv_thr

	Conv2D<HEIGHT,WIDTH,CHANNEL,FILTER,KERNEL,int_t<4,CHANNEL>,win_t<int_t<4,CHANNEL>>> conv;
	MaxPool2x2<HEIGHT,WIDTH,CHANNEL,int_t<4,CHANNEL>> maxpool;

	read_input<12,12,16>(in, even_buf);
	conv.read(16, 16, weight, threshold, conv_wi, conv_thr);
	conv.compute(12, 12, 16, 16, conv_wi, conv_thr, even_buf, odd_buf);
	maxpool.compute(8, 8, 16, odd_buf, even_buf);
	write_result<4,4,16>(out, even_buf);
}
