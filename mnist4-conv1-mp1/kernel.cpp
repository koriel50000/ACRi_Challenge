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

template <typename T>
using win_t = hls::vector<T, KERNEL * KERNEL>;

template <int N>
using int_t = hls::vector<int, N>;

template <typename T>
using fifo = hls::stream<T>;

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
	//T v0_;

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
			} else {
				val = 0;
			}
			if (i < (w + PD) * (KN - 1) - PD) {
				linebuf_.insert_linebuf(val);
			} else {
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
			if (xy == 0) {
				printf("conv window[] y=%d x=%d z=%d\n", KN, KN, c);
				int ptr = 0;
				for (int y = 0; y < KN; y++) {
					for (int x = 0; x < KN; x++) {
						int_t<16> pval = val[ptr++];
						printf("[ ");
						for (int z = 0; z < c; z++) {
							printf("%d ", pval[z]);
						}
						printf("] ");
					}
					printf("\n");
				}
				printf("\n");
			}
			T oval;
			for (int z = 0; z < f; z++) {
				int16_t acc = 0;
				for (int k = 0; k < KN * KN; k++) {
					for (int i = 0; i < c; i++) {
						acc += val[k][i] * wi[z * KN * KN + k][i];
					}
				}
				int m = 0;
				for (int n = 0; n < 3; n++) {
					if (acc >= thr[n]) {
						m = n + 1;
					}
				}
				oval[z] = m;
			}
			outb[xy] = oval;
		}
	}
public:
	//Conv2D(T v0 = 0) : v0_(v0) {}
	
	void read(const int f, const int kn, const int c, const int weight[], const int threshold[],
		T wi[], int thr[])
	{
		int ptr = 0;
		for (int j = 0; j < f * kn * kn; j++) {
			T val;
			for (int i = 0; i < c; i++) {
				val[i] = weight[ptr++];
			}
			wi[j] = val;
		}

		for (int i = 0; i < THRESHOLD; i++) {
			thr[i] = threshold[i];
		}

		printf("read wi[] y=%d x=%d z=%d\n", f, kn * kn, c);
		ptr = 0;
		for (int y = 0; y < f; y++) {
			for (int x = 0; x < kn * kn; x++) {
				int_t<16> pval = wi[ptr++];
				printf("[ ");
				for (int z = 0; z < c; z++) {
					printf("%d ", pval[z]);
				}
				printf("] ");
			}
			printf("\n");
		}
		printf("\n");

		printf("read thr[%d]\n", THRESHOLD);
		for (int i = 0; i < THRESHOLD; i++) {
			printf(" %d", thr[i]);
		}
		printf("\n");
	}

	void compute(const int h, const int w, const int c, const int f, const T wi[], const int thr[],
		const T inb[], T outb[])
	{
		fifo<WT> pips("pipe_fifo");

		printf("compute inb[] y=%d w=%d z=%d\n", h, w, c);
		int ptr = 0;
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int_t<16> pval = outb[ptr++];
				printf("[ ");
				for (int z = 0; z < c; z++) {
					printf("%d ", pval[z]);
				}
				printf("] ");
			}
			printf("\n");
		}
		printf("\n");
	
#pragma HLS dataflow
		windowize(h, w, inb, pips);
		conv(h, w, c, f, wi, thr, outb, pips);

		printf("compute outb[] y=%d x=%d z=%d\n", h - KN + 1, w - KN + 1, f);
		ptr = 0;
		for (int y = 0; y < h - KN + 1; y++) {
			for (int x = 0; x < w - KN + 1; x++) {
				int_t<16> pval = outb[ptr++];
				printf("[ ");
				for (int z = 0; z < f; z++) {
					printf("%d ", pval[z]);
				}
				printf("] ");
			}
			printf("\n");
		}
		printf("\n");	
	}
};

template <int H, int W, int C>
void read_input(const int in[H * W * C], int_t<16> inb[H * W]) {
	int ptr = 0;
	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS unroll factor=W skip_exit_check
		int_t<16> val;
		for (int z = 0; z < C; z++) {
			val[z] = in[ptr++];
		}
		inb[xy] = val;
	}

	printf("read_input H=%d W=%d C=%d\n", H, W, C);
	ptr = 0;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
			int_t<16> pval = inb[ptr++];
			printf("[ ");
			for (int z = 0; z < C; z++) {
				printf("%d ", pval[z]);
			}
			printf("] ");
		}
		printf("\n");
	}
	printf("\n");
}

template <int H, int W, int C>
void write_result(int out[H * W * C], const int_t<16> outb[H * W]) {
	int ptr = 0;
	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS pipeline
		int_t<16> val = outb[xy];
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			out[ptr++] = val[z];
		}
	}

	printf("write_result H=%d W=%d C=%d\n", H, W, C);
	ptr = 0;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
			int_t<16> pval = outb[ptr++];
			printf("[ ");
			for (int z = 0; z < C; z++) {
				printf("%d ", pval[z]);
			}
			printf("] ");
		}
		printf("\n");
	}
	printf("\n");
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

	static int_t<16> even_buf[HEIGHT * WIDTH];
	static int_t<16> odd_buf[OHEIGHT * OWIDTH];
#pragma HLS array_partition variable=even_buf cyclic factor=WIDTH
#pragma HLS array_partition variable=odd_buf cyclic factor=OWIDTH

	static int_t<16> conv_wi[FILTER * KERNEL * KERNEL];
	static int conv_thr[THRESHOLD];
#pragma HLS array_partition variable=conv_wi cyclic factor=KERNEL*KERNEL
#pragma HLS array_partition variable=conv_thr

	Conv2D<12,12,16,5,int_t<16>,win_t<int_t<16>>> conv;

	read_input<12,12,16>(in, even_buf);
	conv.read(16, 5, 16, weight, threshold, conv_wi, conv_thr);
	conv.compute(12, 12, 16, 16, conv_wi, conv_thr, even_buf, odd_buf);
	write_result<8,8,16>(out, odd_buf);
}
