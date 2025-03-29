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
#include <hls_math.h>

const int MAX_WIDTH = 32;
const int MAX_HEIGHT = 32;

const int WIDTH = 24;
const int HEIGHT = 24;
const int CHANNEL = 16;

const int OWIDTH = WIDTH / 2;
const int OHEIGHT = HEIGHT / 2;

using uint4_t = ap_uint<4>;

template <typename T>
using fifo = hls::stream<T>;

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
//#pragma HLS pipeline
			T val1 = inb[ptr++];
			T val2 = inb[ptr++];
			T oval;
			maxpool(val1, val2, oval);
			pips.write(oval);
		}
	}

	void compute_v(const int oh, const int ow, T outb[], fifo<T>& pips) {
		T buf[W / 2];
#pragma HLS array_partition variable=buf

		int ptr = 0;
		for (int y = 0; y < oh; y++) {
//#pragma HLS pipeline
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

template<int H, int W, int C>
void read_input(const int in[H * W * C], int_t<4,C> inb[H * W]) {
	int ptr = 0;
	for (int xy = 0; xy < H * W; xy++) {
//#pragma HLS pipeline
		int_t<4,C> val;
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			val[z] = in[ptr++];
		}
		inb[xy] = val;
	}
}

template<int H, int W, int C>
void write_result(int out[H * W * C], int_t<4,C> outb[H * W]) {
	int ptr = 0;
	for (int xy = 0; xy < H * W; xy++) {
//#pragma HLS pipeline
		int_t<4,C> val = outb[xy];
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			out[ptr++] = val[z];
		}
	}
}

void kernel(int in[HEIGHT * WIDTH * CHANNEL],
	int out[OHEIGHT * OWIDTH * CHANNEL])
{
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHANNEL
#pragma HLS array_partition variable=out cyclic factor=CHANNEL

	static int_t<4,CHANNEL> even_buf[HEIGHT * WIDTH];
	static int_t<4,CHANNEL> odd_buf[OHEIGHT * OWIDTH];
#pragma HLS array_partition variable=even_buf cyclic factor=WIDTH
#pragma HLS array_partition variable=odd_buf cyclic factor=OWIDTH

	MaxPool2x2<int_t<4,CHANNEL>,MAX_HEIGHT,MAX_WIDTH,CHANNEL> maxpool;

//#pragma HLS pipeline

	read_input<HEIGHT,WIDTH,CHANNEL>(in, even_buf);
	maxpool.compute(HEIGHT, WIDTH, even_buf, odd_buf);
	write_result<OHEIGHT,OWIDTH,CHANNEL>(out, odd_buf);
}
