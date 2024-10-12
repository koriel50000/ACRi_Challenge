//#include <stdio.h>
#include "kernel.hpp"
#include "params.hpp"

int16_t muladd(int_t<4,4> vu, int_t<4,4> wu) {
	int16_t acc = 0;
	for (int i = 0; i < 4; i++) {
		int4_t v = vu[i];
		int4_t w = wu[i];
		int sign = (v[3] ^ w[3] == 1) ? -1 : 1;
		acc += sign * v[2,0] * w[2,0];
	}
	return acc;
}

template <int M>
int4_t batch_norm(const int16_t acc, const int threshold[M]) {
	int16_t m = 7 - M;
	for (int i = 0; i < M; i++) {
		if (acc >= threshold[i]) {
			m = i + 1 + (7 - M);
		}
	}
	return m; // TODO sign(1):(3)
}

template <int ROWS, int COLS, typename T, typename WT>
class Window {
private:
	WT buf;
public:
	void shift_pixels_left() {
		for (int i = 0; i < ROWS * COLS - 1; i++) {
			buf[i] = buf[i + 1];
		}
	}

	void insert_right_col(const T value[ROWS]) {
		for (int i = 0; i < ROWS; i++) {
			int idx = (i + 1) * COLS - 1;
			buf[idx] = value[i];
		}
	}

	WT& get_buf() {
		return buf;
	}
};

template <int W, int KN, typename T, typename WT>
class LineBuffer {
private:
	hls::vector<T, (KN - 1) * W> buf;
	Window<KN, KN, T, WT> window;

	void shift_pixels_up() {
		for (int i = 0; i < (KN - 1) * W - 1; i++) {
			buf[i] = buf[i + 1];
		}
	}

	void insert_bottom_row(T value) {
		buf[(KN - 1) * W - 1] = value;
	}

	void get_col(T value[KN - 1]) {
		for (int i = 0; i < KN - 1; i++) {
			value[i] = buf[i * W];
		}
	}
public:
	void insert_linebuf(const T v) {
		shift_pixels_up();
		insert_bottom_row(v);
	}

	void slide_window(const T v) {
		T rows[KN];

		get_col(rows);
		rows[KN - 1] = v;
		shift_pixels_up();
		insert_bottom_row(v);

		window.shift_pixels_left();
		window.insert_right_col(rows);
	}

	WT& get_window() {
		return window.get_buf();
	}
};

template <int H, int W, int C, int KN, int F, int M>
class Conv2D {
public:
	template <typename T, typename OT>
	void compute(fifo<win_t<T,KN*KN>>& ins, fifo<OT>& outs,
		const T weight[F][KN*KN], const int threshold[F][M])
	{
		for (int xy = 0; xy < H * W; xy++) {
			win_t<T,KN*KN> val = ins.read();
			OT oval;
			for (int z = 0; z < F; z++) {
				int16_t acc = 0;
				for (int k = 0; k < KN * KN; k++) {
					T v = val[k];
					T w = weight[z][k];
					acc += muladd(v, w);
				}
				//printf("%d ", acc);
				oval[z] = batch_norm<M>(acc, threshold[z]);
			}
			//printf("\n");
			outs.write(oval);
		}
	}
};

template <int H, int W, typename T>
void read_input(const int in[H * W], fifo<win_t<T,3*3>>& ins) {
	// kernel_size: 3, stride: 2, padding: 1
	LineBuffer<W + 2, 3, T, win_t<T,3*3>> linebuf;

	int ptr = 0;
	T v0 = I3(0x000);
	for (int x = 0; x < W + 2; x++) {
		linebuf.insert_linebuf(v0);
	}
	linebuf.insert_linebuf(v0);
	for (int x = 1; x < W + 1; x++) {
		int c = in[ptr++];
		T v = I3(c);
		linebuf.insert_linebuf(v);
	}
	linebuf.insert_linebuf(v0);

	for (int y = 2; y < H + 1; y++) {
		linebuf.slide_window(v0);
		for (int x = 1; x < W + 1; x++) {
			int c = in[ptr++];
			T v = I3(c);
			linebuf.slide_window(v);

			if (y % 2 == 0 && x % 2 == 0 && 2 <= x && x < W + 1) {
				win_t<T,3*3> oval = linebuf.get_window();
				ins.write(oval);
			}
		}
		linebuf.slide_window(v0);
	}
}

template <typename T>
void write_result(fifo<T>& outs) {
	for (int y = 0; y < 320; y++) {
		for (int x = 0; x < 320; x++) {
			T val = outs.read();
			//printf("[ ");
			for (int z = 0; z < 16; z++) {
				//printf("%d ", val[z]);
			}
			//printf("]\n");
		}
	}
}

const int WIDTH = 640;
const int HEIGHT = 640;

void kernel(int in[HEIGHT * WIDTH],
	int out_obj_8[6400 * 1], int out_cls_8[6400 * 1],
	int out_bbox_8[6400 * 4], int out_kps_8[6400 * 10],
	int out_obj_16[1600 * 1], int out_cls_16[1600 * 1],
	int out_bbox_16[1600 * 4], int out_kps_16[1600 * 10],
	int out_obj_32[400 * 1], int out_cls_32[400 * 1],
	int out_bbox_32[400 * 4], int out_kps_32[400 * 10])
{
	fifo<win_t<int_t<4,4>,3*3>> ins("input_fifo");
	fifo<int_t<4,16>> pips1("pipe_fifo1");

	Conv2D<320, 320, 4, 3, 16, 7> backbone_model0_conv1;

	read_input<640, 640, int_t<4,4>>(in, ins);
	backbone_model0_conv1.compute<int_t<4,4>, int_t<4,16>>(ins, pips1,
		backbone_model0_conv1_weight, // [16][9]
		backbone_model0_relu1_threshold); // [16][7]
	write_result<int_t<4,16>>(pips1);
}
