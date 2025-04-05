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

const int WIDTH = 12;
const int HEIGHT = 12;
const int CHANNEL = 16;

const int FILTER = 16;
const int KERNEL = 5;
const int THRESHOLD = 3;

const int OWIDTH = WIDTH - KERNEL + 1;
const int OHEIGHT = HEIGHT - KERNEL + 1;

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

	for (int i = 0; i < n; i++) {
//#pragma HLS unroll
		t[i] = mul(vu[i], wi[i]);
	}

	for (int d = 1; d < n; d *= 2) {
		for (int i = 0; i < n; i += d * 2) {
//#pragma HLS unroll
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

	void shift_pixels_up() {
#pragma HLS inline
		for (int i = 0; i < W * (KN - 1) - 1; i++) {
#pragma HLS unroll
			buf_[i] = buf_[i + 1];
		}
	}

	void insert_bottom_row(T value) {
#pragma HLS inline
		buf_[W * (KN - 1) - 1] = value;
	}

	void get_col(T value[KN - 1]) {
#pragma HLS inline
		for (int i = 0; i < KN - 1; i++) {
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

template <int H, int W, int C, int KN, typename T, typename WT, int PD = 0, int ST = 1>
class Conv2D {
private:
	LineBuffer<W + PD, KN, T, WT> linebuf_;
	T v0_;

	void windowize(const int h, const int w, const T inb[], fifo<WT>& pips) {
		int x = 0 - (KN - 1);
		int y = 0 - (KN - 1);
		int ptr = 0;
		for (int i = 0; i < (w + PD) * (h + PD * 2) + PD; i++) {
#pragma HLS pipeline
			T val;
			if (0 - (KN - 1) + PD <= x && x < w - (KN - 1) + PD
				&& 0 - (KN - 1) + PD <= y && y < h - (KN - 1) + PD)
			{
				val = inb[ptr++];
			}
			else {
				val = v0_;
			}
			if (i < (w + PD) * (KN - 1) - PD) {
				linebuf_.insert_linebuf(val);
			}
			else {
				linebuf_.slide_window(val);
			}
			if (0 <= x && 0 <= y && x % ST == 0 && y % ST == 0) {
				WT oval = linebuf_.get_window();
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
		for (int xy = 0; xy < (h - KN + 1) * (w - KN + 1); xy++) {
#pragma HLS pipeline
			WT val = pips.read();
			T oval;
			for (int z = 0; z < f; z++) {
				int16_t acc = 0;
				for (int k = 0; k < KN * KN; k++) {
					acc += muladd<C>(c, val[k], wi[z * KN * KN + k]);
				}
				oval[z] = batch_norm(acc, thr, true);
			}
			outb[xy] = oval;
		}
	}
public:
	Conv2D(T v0 = 0) : v0_(v0) {}
	
	void read(const int f, const int kn, const int c, const int weight[], const int threshold[],
		T wi[], int thr[])
	{
		int ptr = 0;
		for (int j = 0; j < f * kn * kn; j++) {
			T val;
			for (int i = 0; i < c; i++) {
				val[i] = (weight[ptr++] << 2) & 0xf;
			}
			wi[j] = val;
		}

		for (int i = 0; i < THRESHOLD; i++) {
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

template <typename T, int H, int W, int C>
class MaxPool2x2 {
private:
	void maxpool(const T v1, const T v2, T& ov) {
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			ov[z] = (v1[z] > v2[z]) ? v1[z] : v2[z];
		}
	}

	void compute_h(const int h, const int w, const T inb[], fifo<T>& pips) {
		int ptr = 0;
		for (int i = 0; i < h * w / 2; i++) {
#pragma HLS pipeline
			T val1 = inb[ptr++];
			T val2 = inb[ptr++];
			T oval;
			maxpool(val1, val2, oval);
			pips.write(oval);
		}
	}

	void compute_v(const int oh, const int ow, T outb[], fifo<T>& pips) {
		static T buf[W / 2];
#pragma HLS array_partition variable=buf

		int ptr = 0;
		for (int y = 0; y < oh; y++) {
#pragma HLS pipeline
			for (int x = 0; x < ow; x++) {
				buf[x] = pips.read();
			}
			for (int x = 0; x < ow; x++) {
				T val1 = buf[x];
				T val2 = pips.read();
				T oval;
				maxpool(val1, val2, oval);
				outb[ptr++] = oval;
			}
		}
	}

public:
	void compute(const int h, const int w, const T inb[], T outb[]) {
		fifo<T> pips("pipe_fifo");

#pragma HLS dataflow
		compute_h(h, w, inb, pips);
		compute_v(h / 2, w / 2, outb, pips);
	}
};

template <int H, int W, int C>
void read_input(const int in[H * W * C], int_t<4,16> inb[H * W]) {

	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS unroll factor=W skip_exit_check
		int_t<4,16> val;
		for (int z = 0; z < C; z++) {
			val[z] = in[xy];
		}
		inb[xy] = val;
	}
}

template <int H, int W, int C>
void write_result(int out[H * W * C], const int_t<4,16> outb[H * W]) {
	int ptr = 0;
	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS pipeline
		int_t<4,16> val = outb[xy];
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			out[ptr++] = val[z];
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
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHANNEL
#pragma HLS array_partition variable=out cyclic factor=FILTER

	static int_t<4,16> even_buf[HEIGHT * WIDTH];
	static int_t<4,16> odd_buf[OHEIGHT * OWIDTH];
#pragma HLS array_partition variable=even_buf cyclic factor=WIDTH
#pragma HLS array_partition variable=odd_buf cyclic factor=OWIDTH

	static int_t<4,16> conv_wi[FILTER * KERNEL * KERNEL];
	static int conv_thr[7] = { 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff, 0x7fff };
#pragma HLS array_partition variable=conv_wi cyclic factor=KERNEL*KERNEL
#pragma HLS array_partition variable=conv_thr

	Conv2D<12,12,16,5,int_t<4,16>,win_t<int_t<4,16>>> conv;

	int v[] = { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, };
	int w[] = { 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, };
	int_t<4,16> vu;
	int_t<4,16> wi;
	int16_t expected = 0;
	for (int i = 0; i < 16; i++) {
		expected += v[i] * w[i];
		vu[i] = v[i];
		wi[i] = (w[i] << 2) & 0xf;
	}
	int16_t actual = muladd<16>(16, vu, wi);
	printf("expected=%d actual=%d\n", expected, actual);

	read_input<12,12,16>(in, even_buf);
	conv.read(16, 5, 16, weight, threshold, conv_wi, conv_thr);
	conv.compute(12, 12, 16, 16, conv_wi, conv_thr, even_buf, odd_buf);
	write_result<8,8,16>(out, odd_buf);
}
