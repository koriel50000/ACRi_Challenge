/*
 * 4bit量子化および演算回路再利用の検証
 * ・weightを1bit符号＋3bit指数部の4bitで表現(0,1,2,4,8,16,32,64,NA,-1,-2,-4,-8,-16,-32,-64) * scale
 * ・バッチ正規化後のactivationを1bit符号+2bit指数部+1bit仮数部の4bitで表現
 *   (0,0.25,0.5,0.75,1.0,1.5,2.0,3.0, NA,-0.25,-0.5,-0.75,-1.0,-1.5,-2.0,-3.0)
 * ・乗算は符号なし3bitの掛け算を、6入力LUTが6個のテーブル参照で計算?もしくはシフト?
 * ・演算回路は最大サイズのConv,Maxpoolを用意し、引数で行列サイズを指定して再利用(範囲外は0埋め?)
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
const int THRESHOLD = 7;

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

uint4_t mul64(const uint6_t i) {
	static const uint4_t table[] = {
0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 1, 1, 1, 1, 1,
0, 1, 1, 2, 2, 2, 2, 2,
0, 1, 2, 3, 3, 3, 3, 3,
0, 1, 2, 4, 4, 4, 4, 4,
0, 2, 3, 6, 6, 6, 6, 6,
0, 2, 4, 8, 8, 8, 8, 8,
0, 3, 6, 12, 12, 12, 12, 12,
	};
	return table[i];
}

int16_t mul(const uint4_t v, const uint4_t w) {
	int16_t oval = mul64((v(2, 0), w(2, 0))) << ((w(1, 0) + 1) & (0 - w[2]));
	return (v[3] ^ w[3]) == 1 ? -oval : oval;
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

	void conv(const int h, const int w, const int c, const int f, const T wi[], const int thr[][THRESHOLD],
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
					oval[j] = batch_norm(acc, thr[j], true);
				}
				outb[y * WIDTH + x] = oval;
			}
		}
	}
public:
	void read(const int c, const int f, const int weight[], const int threshold[][THRESHOLD],
	    T wi[], int thr[][THRESHOLD])
	{
		int ptr = 0;
		for (int j = 0; j < F; j++) {
			if (j >= f) break;
			for (int k = 0; k < KN * KN; k++) {
#pragma HLS pipeline
				T val;
				for (int i = 0; i < C; i++) {
#pragma HLS unroll
					if (i >= c) break;
					val[i] = weight[ptr++] & 0xf;
				}
				wi[j * KN * KN + k] = val;
			}

    		for (int i = 0; i < THRESHOLD; i++) {
#pragma HLS unroll
	    		thr[j][i] = threshold[j][i];
		    }
		}

	}

	void compute(const int h, const int w, const int c, const int f,
	    const T wi[], const int thr[][THRESHOLD], const T inb[], T outb[])
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
					uint4_t val = weight[ptr++] & 0xf;
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
				val[z] = in[ptr++] * 8 - 4;
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
I4(0xa), I4(0x6), I4(0x5), I4(0xc), I4(0xb), I4(0x5), I4(0x5), I4(0x5), I4(0xd), I4(0xb), I4(0x5), I4(0x3), I4(0x1), I4(0x4), I4(0xc), I4(0x5), I4(0xa), I4(0xc), I4(0xd), I4(0xe), I4(0x5), I4(0xa), I4(0xc), I4(0xd), I4(0xe),
I4(0xd), I4(0xb), I4(0x4), I4(0xa), I4(0xd), I4(0xc), I4(0xc), I4(0xe), I4(0xd), I4(0xe), I4(0xd), I4(0xd), I4(0x9), I4(0xb), I4(0xd), I4(0xe), I4(0xb), I4(0x1), I4(0x3), I4(0x3), I4(0x5), I4(0x1), I4(0x5), I4(0xb), I4(0x5),
I4(0x2), I4(0xc), I4(0x5), I4(0x5), I4(0x3), I4(0x6), I4(0x3), I4(0x5), I4(0xd), I4(0xd), I4(0x6), I4(0x4), I4(0xb), I4(0xc), I4(0xb), I4(0xc), I4(0x4), I4(0x5), I4(0x4), I4(0xe), I4(0xd), I4(0x5), I4(0x5), I4(0x4), I4(0xc),
I4(0x3), I4(0xa), I4(0x3), I4(0x4), I4(0x4), I4(0x5), I4(0x4), I4(0x4), I4(0x1), I4(0x5), I4(0xd), I4(0x2), I4(0x2), I4(0x5), I4(0xd), I4(0xd), I4(0x3), I4(0x4), I4(0xd), I4(0xf), I4(0xd), I4(0xf), I4(0xf), I4(0xe), I4(0x1),
I4(0xc), I4(0xb), I4(0x2), I4(0x5), I4(0xe), I4(0xd), I4(0x9), I4(0x9), I4(0x6), I4(0x5), I4(0x3), I4(0x3), I4(0x2), I4(0xc), I4(0x4), I4(0x5), I4(0xb), I4(0xd), I4(0x4), I4(0x5), I4(0x6), I4(0xb), I4(0x0), I4(0x6), I4(0x7),
I4(0x6), I4(0xe), I4(0xe), I4(0xf), I4(0xe), I4(0xc), I4(0xf), I4(0xc), I4(0x4), I4(0x4), I4(0xe), I4(0xd), I4(0x5), I4(0x3), I4(0x3), I4(0xd), I4(0x1), I4(0x3), I4(0x2), I4(0x5), I4(0x9), I4(0x0), I4(0x2), I4(0x1), I4(0x0),
I4(0x4), I4(0xa), I4(0xd), I4(0xb), I4(0xd), I4(0x0), I4(0xa), I4(0xd), I4(0xd), I4(0xe), I4(0xc), I4(0xc), I4(0xd), I4(0x5), I4(0x4), I4(0x5), I4(0xc), I4(0x5), I4(0x4), I4(0x6), I4(0x5), I4(0x3), I4(0x4), I4(0x6), I4(0xd),
I4(0xa), I4(0x5), I4(0x5), I4(0xb), I4(0x5), I4(0x4), I4(0xb), I4(0xd), I4(0xd), I4(0x6), I4(0xc), I4(0x9), I4(0x5), I4(0x9), I4(0x3), I4(0xb), I4(0x4), I4(0x6), I4(0x6), I4(0x4), I4(0x9), I4(0x6), I4(0xc), I4(0xd), I4(0xe),
I4(0x5), I4(0x5), I4(0xa), I4(0xd), I4(0xc), I4(0x4), I4(0x6), I4(0x6), I4(0x5), I4(0x3), I4(0xb), I4(0x4), I4(0x6), I4(0x6), I4(0x3), I4(0xd), I4(0xd), I4(0xc), I4(0x4), I4(0x4), I4(0xe), I4(0xe), I4(0xd), I4(0xc), I4(0x4),
I4(0xc), I4(0x3), I4(0xa), I4(0x1), I4(0x4), I4(0xd), I4(0xc), I4(0x2), I4(0x3), I4(0x5), I4(0xd), I4(0xe), I4(0xc), I4(0xc), I4(0x5), I4(0x5), I4(0xc), I4(0xe), I4(0xc), I4(0x2), I4(0x5), I4(0x4), I4(0x9), I4(0xd), I4(0x9),
I4(0xb), I4(0xd), I4(0x5), I4(0x0), I4(0x4), I4(0xe), I4(0x5), I4(0x7), I4(0xb), I4(0xc), I4(0x3), I4(0x7), I4(0x0), I4(0xd), I4(0xd), I4(0x6), I4(0x5), I4(0xd), I4(0xb), I4(0xd), I4(0x3), I4(0x2), I4(0xd), I4(0x4), I4(0x4),
I4(0x4), I4(0xd), I4(0xe), I4(0xd), I4(0x3), I4(0x5), I4(0xf), I4(0xc), I4(0x3), I4(0x5), I4(0xf), I4(0xd), I4(0xc), I4(0x4), I4(0x5), I4(0xe), I4(0xc), I4(0x4), I4(0x4), I4(0x4), I4(0xe), I4(0x4), I4(0xc), I4(0x0), I4(0x0),
I4(0x2), I4(0x4), I4(0x5), I4(0x5), I4(0x4), I4(0xd), I4(0x9), I4(0xa), I4(0x4), I4(0x3), I4(0x1), I4(0xc), I4(0xc), I4(0xf), I4(0xe), I4(0xb), I4(0xc), I4(0xd), I4(0x3), I4(0xb), I4(0x3), I4(0x2), I4(0x5), I4(0x5), I4(0x4),
I4(0x2), I4(0xc), I4(0xb), I4(0x2), I4(0x5), I4(0x2), I4(0xc), I4(0xd), I4(0xd), I4(0x6), I4(0xb), I4(0x0), I4(0xe), I4(0xb), I4(0x6), I4(0x0), I4(0xb), I4(0x9), I4(0x7), I4(0xa), I4(0x5), I4(0x5), I4(0xb), I4(0x4), I4(0xd),
I4(0xd), I4(0x4), I4(0x4), I4(0x4), I4(0x1), I4(0x3), I4(0x3), I4(0x9), I4(0xa), I4(0xd), I4(0x9), I4(0xe), I4(0xe), I4(0x0), I4(0x5), I4(0x4), I4(0x5), I4(0xa), I4(0x4), I4(0x5), I4(0xd), I4(0xb), I4(0xb), I4(0x3), I4(0xf),
I4(0xa), I4(0x0), I4(0xd), I4(0xe), I4(0xc), I4(0x3), I4(0x4), I4(0x4), I4(0xd), I4(0xe), I4(0xa), I4(0x4), I4(0x5), I4(0xd), I4(0xe), I4(0x2), I4(0x5), I4(0x5), I4(0x4), I4(0xd), I4(0xa), I4(0x9), I4(0x5), I4(0x6), I4(0x4),
	};
	static int conv0_thr[16][7] = {
{ 41, 57, 73, 90, 122, 154, 218, },
{ 164, 170, 175, 180, 191, 202, 223, },
{ 51, 67, 83, 99, 130, 162, 225, },
{ 194, 202, 211, 219, 235, 252, 285, },
{ -29, 0, 27, 55, 112, 169, 283, },
{ 221, 230, 238, 247, 264, 281, 315, },
{ 30, 46, 62, 78, 111, 143, 208, },
{ 39, 57, 75, 93, 129, 165, 236, },
{ 91, 103, 115, 128, 153, 178, 227, },
{ 88, 94, 99, 104, 115, 126, 147, },
{ 135, 148, 160, 173, 198, 224, 274, },
{ 174, 185, 197, 209, 232, 255, 302, },
{ 73, 80, 87, 94, 108, 122, 151, },
{ 95, 105, 116, 126, 147, 168, 209, },
{ 96, 102, 109, 115, 128, 141, 167, },
{ 55, 70, 85, 100, 131, 162, 223, },
    };
#pragma HLS array_partition variable=conv0_wi cyclic factor=KERNEL*KERNEL
#pragma HLS array_partition variable=conv0_thr

	static data_t conv1_wi[FILTER * KERNEL * KERNEL] = {
I4(0xb430a335ccda405b), I4(0xe53c45441dcb1dbb), I4(0x9ccd453bcde42540), I4(0xc34cd55c3ced3934), I4(0xb55b64c934b415c3),
I4(0x5cdde4553bd044d5), I4(0xebc445eb53231d5d), I4(0xc24cbc4eddadbdec), I4(0xc9db2cb34baa5e55), I4(0xed2c25ad4d5e53e4),
I4(0xe250dd92ec4e5c3a), I4(0xeda45a2b2c5e5d5d), I4(0x63ddc53ccc3d650c), I4(0x4dc33c35be4d6c43), I4(0xdea4cddbd5cd2455),
I4(0xd2c52dddcab3bd64), I4(0x5e5512bdb4556d5c), I4(0x4dc5dbbdd2135443), I4(0x45d4cdebd3554a4e), I4(0x52333dcde5444c3b),
I4(0x2546cdbca64c4469), I4(0xc55a2b4a4513bb52), I4(0xb0cdcdccdb1de56a), I4(0x49c4ce59d1b34b5d), I4(0x65142d30452b5d34),
I4(0x4ccd156cc441d934), I4(0xdabb4a5ece4d44bc), I4(0xa65c5393db5d5dd3), I4(0xbaae5442d56b4cc5), I4(0xcc46cb94556c4d5c),
I4(0x54cbcd56c4d4d5a4), I4(0x5c4d0d4bedcbe2aa), I4(0xccab3c5d22d92ca0), I4(0xc4bd44ac441c0dc5), I4(0xd4554d41354e52cb),
I4(0x5305d1a5bcdbd332), I4(0x535e3d5203ece22c), I4(0x3b1a2ccbdbebc4bd), I4(0x4ac3bc5bd3dc454d), I4(0x36b3d555bbcdcbdc),
I4(0x1c5356dc6b44c915), I4(0xc1cca5dd12c2cc0c), I4(0x4d55cdd4342ae54b), I4(0x6adc54c11444dab4), I4(0x5debd2c5ddcdebc3),
I4(0xd550364da5544bd9), I4(0xcb5d94ec5445033d), I4(0x4b53e4de53d63324), I4(0x444cdac2b2dbd0db), I4(0x3bdcedb5bbd1ccdc),
I4(0xca3c4c4e0a51a4a4), I4(0x1b0d53924b44b434), I4(0x5d4a0c42bca4e3cd), I4(0xcdedad3eecb4e2cc), I4(0xbc4c5dded4bced4d),
I4(0xad1ed93d2424d24d), I4(0x453d49cd45b46995), I4(0x56c555bd34a5525b), I4(0x5c45cba2355ac444), I4(0x93b532d54555ad61),
I4(0x449d1c3bdeb4b346), I4(0x52ce1db45de459e5), I4(0x0bd544cc43dc2955), I4(0x54542355354461b4), I4(0x4544cac355c5536d),
I4(0x30cc53ccbde3ccec), I4(0xd3d3d0a52bc994e1), I4(0xcd1d12a43ca5a5d3), I4(0xfbecdc56bbd14c45), I4(0xe0b63c55ad525d35),
I4(0xde4d5364adda4c23), I4(0x0c945d2cdd1daceb), I4(0x2a3d9bd5320a30d5), I4(0xab5c3adddc4ccc4d), I4(0xedd5dcce5ceedabb),
I4(0xeb41cd624be5b94e), I4(0x1a5b3234b4ccdaa3), I4(0xb30e5cde4433e063), I4(0xc4b566ee25411b51), I4(0x55c403dc55694d03),
I4(0x355b492cabcaed4e), I4(0x4b60549c5509dc4c), I4(0xdc3c4cd3bcb3d425), I4(0xc45444e4515bc4cc), I4(0xc55535cadc4bbbc9),
I4(0xec35b5dc334d11b4), I4(0xc34c45a54dacc1dc), I4(0x4424554353d2b5dd), I4(0x4414b41ddda9c3ed), I4(0x4b52e1c356b34c31),
I4(0x54c314d0b9d245c6), I4(0x50cccb44dd5303c3), I4(0x5591ad6433ec3dd4), I4(0x66d3ec541bec34dd), I4(0x4346ed54d4bd546c),
I4(0x5dd4ce54d6d923e3), I4(0x5dd5ee55d5e3a55a), I4(0xbc92cea4c5cc6ed4), I4(0xb1532c44dce26cdc), I4(0x4becd4b64ddd0ad4),
I4(0x15c5cd941c221d9b), I4(0xcdd53b1154cd5341), I4(0x2cd3ddc36c3a54a4), I4(0x39cbe155aad44cdb), I4(0x49fcdd65d1ec53c2),
I4(0x59d4cc9640c4aa4e), I4(0x34e5d965c43c5dbc), I4(0x3dc4de53ead0a5bc), I4(0xd24cce413bebcbd4), I4(0x453dc24553c9d19c),
I4(0x44c6d20456c33acd), I4(0x3355ed40c4c333bd), I4(0x5ccc49b424d515b2), I4(0x3e3b4a5243bcdbd3), I4(0xaead4b4e3c5451d5),
I4(0xc3652ae435544c5e), I4(0xabc403ce34452c34), I4(0x3b2345cc4436c526), I4(0x593e55b0ac0b35d4), I4(0x1a5cc4dd4454c4c5),
I4(0xe44d34ac6c63d52c), I4(0xd51904eb53c4c444), I4(0xe94c45d12163243c), I4(0xdb4c55acdcd59ad5), I4(0x50eb24cc04a4ecdb),
I4(0x30edadbb2b4dc43c), I4(0xcbe3bfc51dcb524d), I4(0x45b3ed30c521443c), I4(0x4dc26354c33d95c5), I4(0x3dd4dccc9ed435ce),
I4(0x36c4ab44a1535d9d), I4(0x345ba224dce25c2c), I4(0xd1d4ddca2ddb4cc5), I4(0x1d35ccc4b4ebe5ca), I4(0xa454dddd54c5ec2b),
I4(0xdd334544cca3d0ed), I4(0xc36c4dca44c2d1d1), I4(0xdc5256cc53051d35), I4(0xdc4a49ad3a459be5), I4(0xcec3b6ecad421d54),
I4(0xc35bb4de55dc5c44), I4(0x624dc6cdcccb4c55), I4(0x2d5c49ba934d61b4), I4(0xe45e6ccdbc6d6be0), I4(0xeec55543ab625455),
I4(0x43ebcbc4de40551c), I4(0x44d64c30eb45b44d), I4(0x54e5dd55d22a4a54), I4(0x04c2e2a5944b53cb), I4(0xdad94d552d34459a),
I4(0x549b92ddd541a45d), I4(0xcbdcebddb51c3d5d), I4(0xcd3bcdaccd3c44eb), I4(0xcb5cd315d4edaa1d), I4(0x5d5154b3dc69bb50),
I4(0xc5c30ce3ac35361c), I4(0x4dd0dac44dd5c4a5), I4(0x0deced5542cdd45b), I4(0xe0dd4d65bd4cd235), I4(0xbb1d695cc45b0a14),
I4(0xdbb34cbddce2c405), I4(0x55d535430dbcebab), I4(0x53db1364cdce3bc3), I4(0x1ed1344bcdcc4c34), I4(0xbc34945cc4bc0c4c),
I4(0xe45d695eca5aa153), I4(0x4390543ccb4c4344), I4(0x4cbd345dd5b325c5), I4(0x43c53bada1acb3ec), I4(0xc4dcce65cedd4bbd),
I4(0xd39b693bb46d4d4c), I4(0x450551a3a4446265), I4(0x59ed4cabab5aca3b), I4(0x4ecdc34eedc4cc0c), I4(0xbe330eaee399bcae),
I4(0x59edbe545bacd345), I4(0x4b3c595b4bda3dc5), I4(0xcc654e31cad1cdd4), I4(0xdd4d5c52ce3bd4ee), I4(0x445e53b02bd052d0),
I4(0x3443c53b3c33d155), I4(0x5643524cb4954444), I4(0x4b4544d224bd0a30), I4(0xdc19bd3355b4ce5c), I4(0xb543023ea5441bbd),
I4(0x3bc323404d4aada5), I4(0x43c6c3dbaa6d2d3c), I4(0xc954a0dcac445ebe), I4(0xe93bc3d55c54e9ad), I4(0xe9dabdd55456e4ac),
I4(0xc5d23e4355ce5dd5), I4(0xcc33c4b9d4424dd5), I4(0xd5cb5d432a34d3dc), I4(0x45dc55533dc3e5c1), I4(0xc4bc5adc5dd52451),
I4(0x2aebd5cbe4cc4dd9), I4(0xa35443ebcddacdbc), I4(0xdc44565cbcd5e42d), I4(0x55b5556d43a43bdc), I4(0x45bc91445043bc64),
I4(0x45e21325d4dd5c4c), I4(0xd5eda45b5ddd4bc4), I4(0xd1945e9439ab26de), I4(0xda2554363b455c0c), I4(0xd11e2c65adc646ec),
I4(0xc5c6dc244b4c645d), I4(0xcde54e2dcc4d5e3b), I4(0x1ab40dbccdd54ebd), I4(0xccd64de35d9d5e34), I4(0xdc1b33bdcddcdecd),
I4(0x544434bdd5624e53), I4(0x3aa55cd9d46cdd9e), I4(0xcb11d345465b51bb), I4(0x2cc665b9355cdc4d), I4(0x9b3c5a294b4c4bbd),
I4(0x4c4c935eccdc5d3b), I4(0xbbaacdc4dd5c0459), I4(0xad2c3d54b952d4dd), I4(0xcce3cb46c55d5d50), I4(0x45c4ed53c3433265),
I4(0x53325d45ba144a64), I4(0x55dbbd4352594454), I4(0x66d5cd34dde4c45c), I4(0x6b9bb341cbdbab5b), I4(0x1dd3d45a4dcb293c),
I4(0x4b5cd5cc459bcdcb), I4(0xdd4b9edb34455d5e), I4(0xda4c49d34245ea44), I4(0xcc9d5dd4534bb240), I4(0xc9e33adc4d5455c4),
I4(0xbea35e3c44cd4e5d), I4(0x3cba54995c45bd54), I4(0x440454d4565d5c93), I4(0xd449c5234ddc6b23), I4(0xb3c2e25cd4cbc45b),
I4(0x4db5bdc3c43d5d31), I4(0xdc4454cdca435433), I4(0xd00b5224a56d6cdc), I4(0xcc3903cd51c55ad0), I4(0xd4cc562d5449cc62),
I4(0xdb4b43dd2d5553cc), I4(0xeb3c34cdcb3c62cd), I4(0xcad5eadecdb39435), I4(0xcc3d4c33ac45d5ba), I4(0x45a46541d44c542b),
I4(0xec3c54feb5c4dd54), I4(0xdc1dd3ed4d4bc4b3), I4(0x03dd2ddeb454d31d), I4(0x5c0a64414eddd5e5), I4(0xcbd9ce5bc4be4bf4),
I4(0x3dac53b6c4bedbdd), I4(0x4c351436bcd34b35), I4(0xc6d50cd54c136055), I4(0xcad5bdd93bbd6250), I4(0xddc63deb5b945d0d),
I4(0xb044bd344dcd4dd5), I4(0xda54c3c4d43b5cc5), I4(0xe0d34d1ddc5cadb3), I4(0xcddcb5bd9d0ceca9), I4(0xdced55ce4d5c0dcb),
I4(0x41dcb4cdab2bbb9b), I4(0x1ddc55cdbc1ceccd), I4(0x5ece4540dc30da5d), I4(0xcece5d4a9b4a030b), I4(0x29bc4a5acd34dd54),
I4(0xc3cec3cd3ce3c632), I4(0xdc5e5ebed4444e2d), I4(0x1b955ac631cd3c05), I4(0xac3c6d4bc43b5425), I4(0x4bce5343d24c2634),
I4(0xcc3c44dedd4c41c5), I4(0x3d924e4dcd39c3ec), I4(0xdd4c5141b9c3d2bc), I4(0x2dcd5b0b2cc5a155), I4(0x319d5aaab4b23355),
I4(0x52ece55dd45344d5), I4(0xcc256ccf1541432d), I4(0x222ca3be554dcd6d), I4(0x43c5b35a3223ed34), I4(0x52edc25cd29d9455),
I4(0x53ade43493d4bb45), I4(0xb5c59ddd3b344d1d), I4(0xb4aa591b4d99dc55), I4(0xd95464be564d5cd6), I4(0x544b04ce4a439955),
I4(0x5d3be454b51dc4c6), I4(0xdd542ab3aa5cb215), I4(0x50cc44dc4b535bb6), I4(0x345c5cdbcd5cc446), I4(0x354553ccd44b4bdb),
I4(0x5a5dbbcb34b42cac), I4(0x4434425441359cd1), I4(0xb35b353ecdd4bdd4), I4(0x5444435c066c05cb), I4(0x1534d2494345dd5c),
I4(0xbcb3bdc9c4d1c5be), I4(0xa012b2ccc3ecbbed), I4(0xd42cbd42e3d50dec), I4(0x4bd5c4453bd3c1ed), I4(0x4042d3a363e2d35d),
I4(0xcdbd655aed3ccdd5), I4(0x4d4cbe440d25b2c2), I4(0xb14acb6bd3e3e4c4), I4(0x54dd533a5d3cd5d5), I4(0xdc5db40cacebcdea),
I4(0xcb2cae4cc3552d5e), I4(0x255c556344c5c2b0), I4(0x555d65bc2541e5e5), I4(0x322a56e55cc3cbd4), I4(0xbce0b1abbc4cc430),
I4(0x0b55cc54053d2d6c), I4(0xc15d43cd4025dc45), I4(0xd4c31aec3ddcb5e5), I4(0x54e394ed0c42333a), I4(0x44c41d54c5445c63),
I4(0x34b5c5d45b5b2451), I4(0xcc0c4492d0d3e1c5), I4(0x2bd444dbc31ce9d3), I4(0x9de3d5ababecea5d), I4(0x5c45c35c241d1d44),
I4(0x56c442cbcc15d1d5), I4(0x5eab6555d292d9e4), I4(0x59110d2ad5db2a3a), I4(0x91c44202459ad4b5), I4(0xcb4a3abdcdcca45c),
I4(0xd25ccbd59ca3d4b4), I4(0xdc4c544c4c5dda44), I4(0x0454569c4e43ade5), I4(0x24a564631445c6d0), I4(0x4da2ce3c2b544351),
I4(0x4993cb4d19b4bb2b), I4(0x5455055db9dcb0bb), I4(0x34adbd45dbccddca), I4(0x0d42da445253cd2d), I4(0xd45cbc93d4e5bc9e),
I4(0xb245a5cb25a042d4), I4(0x6bdbce6bb543a45c), I4(0x5a45346a4604d14c), I4(0x1965c5c44cdd9c55), I4(0xd54bd359541d3c1b),
I4(0xa5c504c13454054b), I4(0x5b4dddc53ced2d1d), I4(0x4ad6b2b9a59414ad), I4(0x6ddbf41ae6d5554b), I4(0xc1b4eebd4ccb7543),
I4(0xe3bb450d5fd514b2), I4(0xe51be5454eec4154), I4(0x099ccac53c5d4d4b), I4(0x46c2dde65454cd3c), I4(0x43d1d0e4e659054d),
I4(0x1c4eb4de34c54d5a), I4(0xb54db5e464394d4c), I4(0x55c3c345d5d53423), I4(0x64e4ed36aadab2ad), I4(0xc6a4eaa654d3455d),
I4(0x239b45cd2dc435d3), I4(0xe54c45d4ad4039e4), I4(0x55401654a0b3c5db), I4(0x55d25b5a3d4593dc), I4(0xc4d3d45b44d2d4c3),
I4(0x424d4c5ddc5dc4d5), I4(0xc95cd50c4b9a94cb), I4(0x435bb5540cc45dce), I4(0xa55c9562d5d2ccdd), I4(0x4a3d45542be659d4),
I4(0x55ddcc42bce4ab2c), I4(0xab35dea5cdcddcb4), I4(0x3a5dc6c5b3d3dead), I4(0xc25a533d4222535c), I4(0xc95c645d11d3ddcd),
I4(0x4b15a4db25cdad44), I4(0xb554354dc95e5c39), I4(0xec5b45c91cd35dca), I4(0xebb445ddce3ac345), I4(0xcd34439a4364d3c3),
I4(0xe35e03443bdc9c4d), I4(0xbdcb533decc3ebcc), I4(0x3a5539db52dae4cc), I4(0xbd3d59ce4c4ddabc), I4(0xce3ea23ccde4dbd9),
I4(0xcdad053cd2b35b5a), I4(0xbd2a2ec5cdb3913d), I4(0x5d5d43c4dc54bb9d), I4(0x4ddca29dd443c5c4), I4(0xb34d5a4c2bc10c44),
I4(0xb4ccb2cbc53c94c4), I4(0xdcdd42423c333b6c), I4(0x043dcc249dccd4b4), I4(0xb5be04ac12dce2cc), I4(0x4a4c245dd54bd444),
I4(0xaddd5da5dcb24334), I4(0xcd9e33340bc4ed45), I4(0xbccd5c0cdddbebd5), I4(0x5bbe0d44c4d4e44c), I4(0x4ccec45a94addb3b),
I4(0x5d5d4400dd4de1da), I4(0x5e5d5544dbdce5d6), I4(0x5c1e655ca0a1caca), I4(0x353d1db3c5b3d55c), I4(0x9d0cbc6441c5446c),
	};
	static int conv1_thr[16][7] = {
{ 187, 227, 267, 307, 386, 466, 626, },
{ 71, 105, 138, 171, 238, 305, 439, },
{ 233, 275, 317, 360, 444, 528, 697, },
{ 468, 512, 555, 599, 686, 774, 948, },
{ 282, 325, 369, 413, 500, 588, 763, },
{ 127, 176, 224, 273, 369, 466, 659, },
{ 3, 50, 98, 145, 240, 335, 525, },
{ 139, 178, 218, 257, 336, 414, 571, },
{ 300, 340, 380, 420, 500, 580, 740, },
{ 281, 317, 353, 389, 461, 533, 678, },
{ -94, -44, 6, 57, 158, 260, 463, },
{ 322, 365, 408, 451, 537, 624, 796, },
{ 238, 284, 329, 374, 465, 556, 737, },
{ 173, 238, 304, 369, 499, 629, 890, },
{ 213, 253, 293, 333, 413, 494, 654, },
{ -295, -252, -208, -165, -78, 9, 183, },
	};
#pragma HLS array_partition variable=conv1_wi cyclic factor=KERNEL*KERNEL
#pragma HLS array_partition variable=conv1_thr

	static data_t mat_wi[CLASS * FLATTEN / CHUNK_SIZE] = {
I4(0xaceddbc42dbacc24), I4(0x242edadcc4564354), I4(0xc42ddc454ca0c454), I4(0xd4ae9c424233c1cc),
I4(0x40c4b3cad5dbdddd), I4(0x42c3b4eca3de40cd), I4(0x2ee6cd9d4c1bdde2), I4(0x354ddc94b245d5b2),
I4(0xb9b34b4943bd0b11), I4(0xdebc4c5d4d2b59c9), I4(0xb91ddc4a3a215193), I4(0x3044a5ba44dec409),
I4(0x24acc5c1c551bac3), I4(0xdc0ccdd45154c3c9), I4(0x52d423329542133c), I4(0xbcbac4dc469d44c2),
I4(0xbd46953c2addbd44), I4(0xa445dc44cd04e4d0), I4(0xb31b3c943dbb2c23), I4(0xc44d3d424e425d3a),
I4(0x541ccb3bc5d5b4c5), I4(0x5e05c50d42d923cd), I4(0x54ccd911d44add3a), I4(0x1cb39cdb4554cdad),
I4(0x1dd555cdbcc4bc31), I4(0xdd1d4400bd9b4535), I4(0x3d44444d554d4055), I4(0x5430bed45cc344d4),
I4(0xcb0cadbbccbbb4cb), I4(0xb23d9e6bcbbbeb3d), I4(0x55addac4dbeb24bd), I4(0x3ee955544ccbe4c3),
I4(0x44cce4ed44ccedcc), I4(0xb4d434eda354ecb4), I4(0x5ea451433dace3cd), I4(0xed5c6a354c0e55e3),
I4(0xc4a22563c4145ca6), I4(0x4e4a54dc5ac4c4b4), I4(0xc4cc4c9c4c129c30), I4(0x34ac00a9d2443eac),
I4(0x1d33db5e33b1bb9e), I4(0xb0eceb3a4b5fbb34), I4(0xc2ce43524ac5a355), I4(0xba2ec4652452dcc5),
I4(0x02533dcccb43310e), I4(0x3c2cacd9454e4314), I4(0x4ec4baae9ccbecbd), I4(0xd3c434c5cd04e4b5),
I4(0x454343d29cc25ccd), I4(0x5b5b54cec0dc54ce), I4(0xc5c4b55db61b534c), I4(0x40c3d94d3cd2dccc),
I4(0xc44dca44b5d55955), I4(0xbbdbc4d32d59cdcc), I4(0x4d244e333db055a4), I4(0x3c33c4dce55bb5cc),
I4(0x3eb4325b5ddba255), I4(0xeb14b524a625e6d2), I4(0x454bcc9ace44ca0b), I4(0x54dc1b21bd4d4ded),
I4(0x45eada5503db3e4b), I4(0x3edcd343b3dbdbd5), I4(0x44ca49b31b95e3a1), I4(0xad1c3cd2a529429e),
I4(0x4c4a4dd65ecd444b), I4(0xbc4cb6a4dc19343c), I4(0xcc4254432c3d4db3), I4(0x33db55aca5acb5c3),
I4(0xcc5a3eccbc9b311d), I4(0x342acdb4b344ddc4), I4(0x55dadd3c2cb10dbf), I4(0x4dd1931c50dd25bd),
I4(0x4ca5322b4dc0b1a4), I4(0xbe56bd41406a444c), I4(0x4dc44b5ec4cdc04b), I4(0xc94d4505b0d05549),
I4(0xc13ed945d9d45e5b), I4(0x3c434cc5bb93dc9b), I4(0xb3551bb3b05abacd), I4(0xc3dbd9e4a1c3d50c),
I4(0x53449d9f2c4db9ce), I4(0xbc4dcb33a45dd104), I4(0xcecc0c3c4e351ba0), I4(0xba3e9253553dd15a),
I4(0xbac5432edd525350), I4(0x2d5d3d543454c353), I4(0x33c3abae1eafc2ee), I4(0xcbec33430514dcb4),
I4(0x15d4add42dc53ceb), I4(0x33345cdbdb1c5499), I4(0x54aa3c54d44cabd0), I4(0x4ac2c55035cd4404),
I4(0x45dd103acd3b4c34), I4(0xbd0d43c39eda4dcc), I4(0x19c439c6b5a944dd), I4(0xbb3094c4d91ab21c),
I4(0x3d0b4cbe24aec42b), I4(0xd1bbdc4245cbfdd0), I4(0x4b43bc3ecd049150), I4(0xdb4c4d45b2ba40eb),
I4(0xc30da3c42330e055), I4(0x3dd23b6444134196), I4(0xcc42bcca4e49354c), I4(0x1bcaa1b4ec542d4d),
I4(0xdc4ac22533bcc5c3), I4(0x24d494023531d05d), I4(0xc45dc552e3b29d4b), I4(0xc394a1495dbc5535),
I4(0x5c35bccdecd2345c), I4(0x4dc994bb54133ce9), I4(0xc5b454452d913c4e), I4(0xcd334ccc4acbd024),
I4(0x4d525cc43ccd54c5), I4(0xb5bad0ea10454e2e), I4(0xad540dcfacdda5d4), I4(0x5bcdfcd5dbb1de5b),
I4(0x15bddcc5dc5b4159), I4(0xdc393dcddca423c3), I4(0xb4d04552b95a5d4d), I4(0xadeba34043abe4c2),
I4(0x54043edc153f534d), I4(0x312524149ccceab5), I4(0x4e33144e134e51cd), I4(0x0d5bdc35b3ccc055),
I4(0xd3dcb339bc35bc0d), I4(0xdb13b9142c3ecb35), I4(0x45baddab42cd34ee), I4(0x32e3e44bce12ed5c),
I4(0x4c3541ab952253df), I4(0xc5ce42b3d4c5b354), I4(0x6b4e4dbee93fc3bd), I4(0x4c2333a4365e403d),
I4(0x15bcb30dcc3355bb), I4(0xd0aecb43bec2ca63), I4(0xb3d454344310a0e0), I4(0xdc54cbb3d913d243),
I4(0x232dbfdd9d3e53ba), I4(0x2ce5c33c649cbddc), I4(0xa3b23be04a2eaaac), I4(0x39d55b923a15e2e5),
I4(0xec3c524bedace4d4), I4(0x4dbde20c5b5d45c5), I4(0x93493bb3ddb05123), I4(0xcb9b1dc4dd26dc54),
I4(0x3dcb351c554a43de), I4(0xd0cb5cd01bc4db6a), I4(0xb34cceb2decc63c4), I4(0x3cd4a4bd54b1d45e),
I4(0xc5ad1db4ddacad65), I4(0x5c24b544344bdcdd), I4(0xdeb2345b4d55dce5), I4(0xbd4fcc0ccbd26c33),
I4(0xe45d1d06dccd5bd5), I4(0x45d3bca34344dc5c), I4(0xbc4d3ced44ce51d4), I4(0xb4e3dc455435ea59),
I4(0xccd303cd5b430a33), I4(0x4dd4033d3adcd3ee), I4(0x31b2c4562355ed6c), I4(0x5ac44cde5bbc53c3),
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
