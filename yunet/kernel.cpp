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
	hls::vector<T, W * (KN - 1)> buf_;
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

template <int H, int W, int KN, typename IT, typename OT, int PD = 0, int ST = 1>
class WindowBuffer {
private:
	LineBuffer<W + PD, KN, IT, OT> linebuf_;
public:
	void pass_through(fifo<IT>& ins, fifo<OT>& outs) {
		int x = 0 - (KN - 1);
		int y = 0 - (KN - 1);
		for (int i = 0; i < (W + PD) * (H + PD * 2) + PD; i++) {
#pragma HLS pipeline
			IT val;
			if (0 - (KN - 1) + PD <= x && x < W - (KN - 1) + PD
				&& 0 - (KN - 1) + PD <= y && y < H - (KN - 1) + PD)
			{
				val = ins.read();
			}
			else {
				val = 0;
			}
			if (i < (W + PD) * (KN - 1) - PD) {
				linebuf_.insert_linebuf(val);
			}
			else {
				linebuf_.slide_window(val);
			}
			if (0 <= x && 0 <= y && x % ST == 0 && y % ST == 0) {
				OT oval = linebuf_.get_window();
				outs.write(oval);
			}
			x++;
			if (x >= W - (KN - 1) + PD * 2) {
				x = 0 - (KN - 1) + PD;
				y++;
			}
		}
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
void read_input(const int in[H * W], fifo<T>& ins) {
	for (int xy = 0; xy < H * W; xy++) {
		T val = in[xy];
		ins.write(val);
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
	fifo<int_t<4,4>> ins("input_fifo");
	fifo<win_t<int_t<4,4>,3*3>> pips1("pipe_fifo1");
	fifo<int_t<4,16>> pips2("pipe_fifo2");

	WindowBuffer<640, 640, 3, int_t<4,4>, win_t<int_t<4,4>,3*3>, 1, 2> backbone_model0_buffer1;
	Conv2D<320, 320, 4, 3, 16, 7> backbone_model0_conv1;

	read_input<640, 640, int_t<4,4>>(in, ins);
	//backbone_model0_buffer1.pass_through(ins, pips1);
	//backbone_model0_conv1.compute<int_t<4,4>, int_t<4,16>>(pips1, pips2,
	//	backbone_model0_conv1_weight, // [16][9]
	//	backbone_model0_relu1_threshold); // [16][7]
	//write_result<int_t<4,16>>(pips2);

}
