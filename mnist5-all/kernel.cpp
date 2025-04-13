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

const int WIDTH = 28;
const int HEIGHT = 28;
const int CHANNEL = 16;
const int FILTER = 16;

const int KERNEL = 5;
const int THRESHOLD = 3;

const int FLATTEN = 256;
const int CLASS = 10;
const int CHUNK_SIZE = 16;

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
	void read(const int c, const int f, const int weight[], const int threshold[], T wi[], int thr[]) {
		int ptr = 0;
		for (int j = 0; j < F; j++) {
			if (j >= f) break;
			for (int k = 0; k < KN * KN; k++) {
#pragma HLS pipeline
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

template <int CL, int FL, int K, int H, int W>
class Dense {
private:
	using IT = int_t<K>;
	using OT = int_t<CL,16>;

	void flatten(const IT mat[CL * FL / K], const IT inb[], fifo<OT>& pips) {
		int ptr = 0;
		for (int y = 0; y < H; y++) {
			for (int x = 0; x < W; x++) {
				IT vu = inb[y * WIDTH + x];
				OT oval;
				for (int i = 0; i < CL; i++) {
#pragma HLS pipeline
					IT wi = mat[ptr++];
					int16_t acc = muladd<K>(K, vu, wi);
					oval[i] = acc;
				}
				pips.write(oval);
			}
		}
	}

	void write_result(int out[CL], fifo<OT>& pips) {
		static int16_t acc[CL];
#pragma HLS array_partition variable=acc
	
		for (int i = 0; i < CL; i++) {
#pragma HLS unroll
			acc[i] = 0;
		}
	
		for (int j = 0; j < FL / K; j++) {
#pragma HLS pipeline
			OT val = pips.read();
			for (int i = 0; i < CL; i++) {
#pragma HLS unroll
				acc[i] += val[i];
			}
		}
	
		for (int i = 0; i < CL; i++) {
#pragma HLS unroll
			out[i] = acc[i];
		}
	}
public:
	void read(const int weight[CL * FL], IT mat[CL * FL / K]) {
		int ptr = 0;
		for (int i = 0; i < CL; i++) {
#pragma HLS pipeline
			for (int j = 0; j < FL / K; j++) {
				for (int k = 0; k < K; k++) {
#pragma HLS unroll
					uint4_t val = (weight[ptr++] << 2) & 0xf;
					mat[j * CL + i][k] = val;
				}
			}
		}
	}

	void compute_and_write_result(int out[CL], const IT mat[CL * FL / K], const IT inb[]) {
		fifo<OT> pips("pipe_fifo");

#pragma HLS dataflow
		flatten(mat, inb, pips);
		write_result(out, pips);
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

using data_t = int_t<CHANNEL>;
using block_data_t = data_t[HEIGHT * WIDTH];
using win_t = hls::vector<data_t, KERNEL * KERNEL>;
template <typename T>
using sob = hls::stream_of_blocks<T>;

void kernel(
	int in[28 * 28 * 1],
	int conv0_weight[16 * 5 * 5 * 1],
	int threshold0[3],
	int conv1_weight[16 * 5 * 5 * 16],
	int threshold1[3],
	int matmul0_weight[10 * 256],
	int out[10])
{
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=16
#pragma HLS array_partition variable=conv0_weight cyclic factor=25
#pragma HLS array_partition variable=threshold0 cyclic factor=3
#pragma HLS array_partition variable=conv1_weight cyclic factor=16
#pragma HLS array_partition variable=threshold1 cyclic factor=3
#pragma HLS array_partition variable=matmul0_weight cyclic factor=16
#pragma HLS array_partition variable=out cyclic factor=10

	static block_data_t even_buf;
	static block_data_t odd_buf;
#pragma HLS array_partition variable=even_buf cyclic factor=WIDTH
#pragma HLS array_partition variable=odd_buf cyclic factor=WIDTH

	static data_t conv0_wi[FILTER * KERNEL * KERNEL] = {
I4(0x0), I4(0xc), I4(0xc), I4(0xc), I4(0xc), I4(0x0), I4(0x0), I4(0xc), I4(0xc), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x4), I4(0x4), I4(0x4), I4(0x4), I4(0x0), I4(0x4), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x0),
I4(0x4), I4(0x4), I4(0x0), I4(0x0), I4(0xc), I4(0x4), I4(0x4), I4(0x0), I4(0xc), I4(0xc), I4(0x4), I4(0x0), I4(0xc), I4(0xc), I4(0xc), I4(0x4), I4(0x0), I4(0xc), I4(0xc), I4(0xc), I4(0x0), I4(0x0), I4(0xc), I4(0xc), I4(0xc),
I4(0xc), I4(0x0), I4(0x0), I4(0x4), I4(0x4), I4(0x0), I4(0x0), I4(0x4), I4(0x4), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0xc), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0xc), I4(0x0),
I4(0x0), I4(0xc), I4(0x0), I4(0x0), I4(0x0), I4(0xc), I4(0x0), I4(0x0), I4(0x4), I4(0x4), I4(0xc), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0xc), I4(0xc), I4(0x4), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x0),
I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0xc), I4(0x0), I4(0xc), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x4), I4(0x4), I4(0xc), I4(0x4), I4(0x4), I4(0x4), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x4), I4(0x4), I4(0x0),
I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x4), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x0),
I4(0x4), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0xc), I4(0xc), I4(0x0), I4(0x0), I4(0x0), I4(0xc), I4(0xc), I4(0xc), I4(0xc),
I4(0x0), I4(0x0), I4(0x0), I4(0xc), I4(0x0), I4(0x0), I4(0xc), I4(0x0), I4(0x0), I4(0x0), I4(0xc), I4(0xc), I4(0xc), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x4), I4(0x4),
I4(0x4), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x4), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0xc), I4(0x0), I4(0xc), I4(0x0), I4(0x0), I4(0xc), I4(0x0), I4(0x0), I4(0x0), I4(0x4),
I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x4), I4(0x4), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0xc),
I4(0x0), I4(0xc), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0xc), I4(0x0), I4(0x4), I4(0x0), I4(0x4), I4(0x0), I4(0xc), I4(0x0), I4(0x0), I4(0x4),
I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x4), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0xc), I4(0x0), I4(0x0), I4(0x0), I4(0x0),
I4(0x0), I4(0x0), I4(0x4), I4(0x4), I4(0x0), I4(0x4), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0xc), I4(0x0), I4(0xc), I4(0xc), I4(0x0), I4(0xc), I4(0xc), I4(0xc), I4(0xc), I4(0x0),
I4(0x4), I4(0x0), I4(0x4), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0xc), I4(0xc), I4(0x0), I4(0x0), I4(0x4), I4(0xc), I4(0xc), I4(0xc), I4(0xc), I4(0xc),
I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0xc), I4(0x0), I4(0x4), I4(0x0), I4(0x0), I4(0xc), I4(0xc), I4(0x0), I4(0x4), I4(0x0), I4(0xc), I4(0xc), I4(0x4), I4(0x4), I4(0x0), I4(0x0), I4(0x0),
I4(0xc), I4(0xc), I4(0x0), I4(0xc), I4(0x0), I4(0xc), I4(0xc), I4(0x0), I4(0xc), I4(0xc), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0xc), I4(0x4), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x0), I4(0x4), I4(0x0), I4(0x4), I4(0x4),
	};
	static int conv0_thr[7] = { 1, 3, 4, 0x7fff, 0x7fff, 0x7fff, 0x7fff };
#pragma HLS array_partition variable=conv0_wi cyclic factor=KERNEL*KERNEL
#pragma HLS array_partition variable=conv0_thr

	static data_t conv1_wi[FILTER * KERNEL * KERNEL] = {
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
	static int conv1_thr[7] = { 3, 9, 14, 0x7fff, 0x7fff, 0x7fff, 0x7fff };
#pragma HLS array_partition variable=conv1_wi cyclic factor=KERNEL*KERNEL
#pragma HLS array_partition variable=conv1_thr

	static data_t mat_wi[CLASS * FLATTEN / CHUNK_SIZE] = {
I4(0x0c00000000000000), I4(0x0000c00cccc00000), I4(0xc0cc0040c4c004c0), I4(0xc0ccc400c4c0c4cc),
I4(0x4c04000000400c00), I4(0x000000004c400000), I4(0x0c000c0c00000c00), I4(0x000000c400000000),
I4(0x00040cc000400000), I4(0x000000c44c040004), I4(0xc00000c00c000000), I4(0x00000c004c004000),
I4(0x04c0c0000000c0cc), I4(0x040c00c0c0c00000), I4(0x0c00000000000c00), I4(0x0000000400000000),
I4(0x0c000c0000000c00), I4(0x400c04c000ccc400), I4(0x00000c0004000000), I4(0x000000c40c040004),
I4(0x00000c00cc00c000), I4(0x0c40004444000c00), I4(0x0400c00c00000000), I4(0x0000000000004004),
I4(0x0c040c004c040c00), I4(0xc00c040000c0c400), I4(0xcc00044c44000000), I4(0x40004400000c0440),
I4(0x040000c0c0000000), I4(0x04000cc0c00c0000), I4(0x44000cc00000c000), I4(0xcc00cc0000040000),
I4(0x0000c00000000000), I4(0x00000c00c0000000), I4(0xcc040c0044044c04), I4(0x04cc0400c0ccc00c),
I4(0xc00c040c000c0c0c), I4(0x0c04440000000440), I4(0x0400000000004000), I4(0x400c4000000c0400),
I4(0x0c040c000c400044), I4(0x000c0c000cccc0cc), I4(0x000c040004c00000), I4(0x00ccc400c0c0000c),
I4(0x00040cc00c400004), I4(0x00000cc40c000000), I4(0x0c440c004c400c00), I4(0xc0ccc44004c00000),
I4(0x000000c0000000c0), I4(0x40040c0000400000), I4(0xc0c00c4c00000c00), I4(0x00044ccc4c000000),
I4(0x0000040000c00000), I4(0x000004c00c044400), I4(0x40044c00004c0400), I4(0x0000004000000000),
I4(0x0c004c0400400000), I4(0xc0c0044c00000000), I4(0x4000c0c000000000), I4(0x00000004c000000c),
I4(0xc000c00cc0000ccc), I4(0x0c400c4444400c00), I4(0xc004000040000c00), I4(0x00040c000c00c004),
I4(0x00004cc440000000), I4(0x00cc0440c0ccc00c), I4(0x000c400400000000), I4(0xc40000cc00000cc0),
I4(0x000c00c0c0000400), I4(0x000000c0c0000000), I4(0x0404c0cc0c0440c4), I4(0x000ccc0cc0c000cc),
I4(0x0c00000040040c00), I4(0x00000000c00000c0), I4(0x0c00000404040004), I4(0x00cc040000ccc00c),
I4(0x400044c40c0c4440), I4(0x04000c0004000000), I4(0xc0000000c0000000), I4(0x0040cc000c00c000),
I4(0x4c440c0000400c04), I4(0x00000c0000c00000), I4(0x0044400000000044), I4(0x000c000000c0c00c),
I4(0x00c0cc00c0000400), I4(0x000c000004c0c000), I4(0x00040cc040400000), I4(0x00cc0000c00000c0),
I4(0xc00400cc0c004c00), I4(0x00cccc00000000cc), I4(0x4c000040040c00c0), I4(0xcc040c0000440000),
I4(0x000000c00c000044), I4(0x000cc00c00c4c00c), I4(0x00cc0000cc0c0000), I4(0x000c040004000000),
I4(0x40440c0000000000), I4(0x0000404c40000000), I4(0x00440c0c0c000c04), I4(0x00cc00c0c000000c),
I4(0xc00c40c040c00000), I4(0x0c00cc040404cc00), I4(0x4000000000000400), I4(0x0000040004000000),
I4(0x000c0c0000000000), I4(0x0000040000000000), I4(0x000c00c000004000), I4(0xcc04000000040c40),
I4(0x000ccc4cc40cc0cc), I4(0x00040c0c00000000), I4(0xcc0000c00c040004), I4(0x0000000cc0c0400c),
I4(0x000c400440000000), I4(0x0000000000000000), I4(0x04c00c00c000c000), I4(0x0000000000000000),
I4(0x004000000c004404), I4(0x00000c000000c0c0), I4(0x000c040000ccc0cc), I4(0xc000cc4c0000ccc0),
I4(0x0040cc0000000004), I4(0xc0004c0c00000000), I4(0x00000c0000000000), I4(0x40000404040c0440),
I4(0xc0c0004cc0000ccc), I4(0x4000040404000400), I4(0xc0c00cc0cc0000c0), I4(0xc000000c0c000c00),
I4(0x00440cc000400000), I4(0x00cc004c00c00c00), I4(0x00004004c00c0400), I4(0x0004000000440c00),
I4(0x000cc0000000000c), I4(0x000c04c000c00040), I4(0xc000040c0c004c00), I4(0x0000000000000040),
I4(0x40ccccc0c0c0c0c0), I4(0xcc444c0c00404c00), I4(0x4040cc0000000400), I4(0xc000040c40000c00),
I4(0x000c000000c00440), I4(0x00000c0404000000), I4(0x04cc000000ccc00c), I4(0x000404000c000040),
I4(0x0000c00c00400cc0), I4(0x004040000c000000), I4(0x00cc0cc0cc0cc40c), I4(0xcc000c0004000c00),
I4(0x0040c00000000400), I4(0x0c00000c40040c00), I4(0xc0c000000004c000), I4(0x0000000000c0000c),
I4(0x400c44c4000cc40c), I4(0x0000000000000004), I4(0x0000000004000cc0), I4(0x000000000c004000),
I4(0x00cc0c0c0000c400), I4(0xc0c0c40004c00cc0), I4(0x0040000000400004), I4(0x0c040c0000004c00),
	};
#pragma HLS array_partition variable=mat_wi cyclic factor=FLATTEN/CHUNK_SIZE

	Conv2D<HEIGHT,WIDTH,CHANNEL,FILTER,KERNEL> conv;
	MaxPool2x2<HEIGHT,WIDTH,CHANNEL> maxpool;
	Dense<CLASS,FLATTEN,CHUNK_SIZE,4,4> matmul0;

	read_input<28,28,1,data_t>(in, even_buf);
	conv.compute(28, 28, 1, 16, conv0_wi, conv0_thr, even_buf, odd_buf);
	maxpool.compute(24, 24, 16, odd_buf, even_buf);
	conv.compute(12, 12, 16, 16, conv1_wi, conv1_thr, even_buf, odd_buf);
	maxpool.compute(8, 8, 16, odd_buf, even_buf);
	matmul0.compute_and_write_result(out, mat_wi, even_buf);
}
