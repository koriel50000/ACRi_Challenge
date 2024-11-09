//#include <stdio.h>
#include "kernel.hpp"
#include "params.hpp"

template <int C>
int16_t muladd(const int_t<4,C> vu, const int_t<4,C> wu) {
	int16_t acc = 0;
	for (int i = 0; i < C; i++) {
		int4_t v = vu[i];
		int4_t w = wu[i];
		acc += v * w;
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
	return m;
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
#pragma HLS array_partition variable=buf_
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

template <int H, int W, int C, int KN, int PD = 0, int ST = 1>
class Conv2D {
public:
	void windowize(fifo<int_t<4,C>>& ins, fifo<win_t<int_t<4,C>,KN*KN>>& outs) {
		LineBuffer<W + PD, KN, int_t<4,C>, win_t<int_t<4,C>,KN*KN>> linebuf_;

		int x = 0 - (KN - 1);
		int y = 0 - (KN - 1);
		for (int i = 0; i < (W + PD) * (H + PD * 2) + PD; i++) {
#pragma HLS pipeline
			int_t<4,C> val;
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
				win_t<int_t<4,C>,KN*KN> oval = linebuf_.get_window();
				outs.write(oval);
			}

			x++;
			if (x >= W - (KN - 1) + PD * 2) {
				x = 0 - (KN - 1) + PD;
				y++;
			}
		}
	}

	template<int OH, int OW, int F, int M>
	void compute(fifo<win_t<int_t<4,C>,KN*KN>>& ins, fifo<int_t<4,F>>& outs,
		const int_t<4,C> weight[F][KN*KN], const int threshold[F][M])
	{
		for (int xy = 0; xy < OH * OW; xy++) {
			win_t<int_t<4,C>,KN*KN> val = ins.read();
			int_t<4,F> oval;
			for (int z = 0; z < F; z++) {
				int16_t acc = 0;
				for (int k = 0; k < KN * KN; k++) {
					int_t<4,C> v = val[k];
					int_t<4,C> w = weight[z][k];
					acc += muladd<C>(v, w);
				}
				//printf("%d ", acc);
				oval[z] = batch_norm<M>(acc, threshold[z]);
			}
			//printf("\n");
			outs.write(oval);
		}
	}
};

template <int H, int W, int C>
class Conv2D<H, W, C, 1> {
public:
	template<int OH, int OW, int F, int M>
	void compute(fifo<int_t<4,C>>& ins, fifo<int_t<4,F>>& outs,
		const int_t<4,C> weight[F][1], const int threshold[F][M])
	{
		for (int xy = 0; xy < OH * OW; xy++) {
			int_t<4,C> val = ins.read();
			int_t<4,F> oval;
			for (int z = 0; z < F; z++) {
				int16_t acc = muladd<C>(val, weight[z][1]);
				//printf("%d ", acc);
				oval[z] = batch_norm<M>(acc, threshold[z]);
			}
			//printf("\n");
			outs.write(oval);
		}
	}
};

template <int H, int W, int C>
void read_input(const int in[H * W], fifo<int_t<4,C>>& ins) {
	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS unroll factor=16 skip_exit_check
		int_t<4,C> val = in[xy];
		ins.write(val);
	}
}

template <int H, int W, int C>
void write_result(int out[16], fifo<int_t<4,C>>& outs) {
	int acc = 0;
	for (int y = 0; y < H; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			int_t<4,C> val = outs.read();
			//printf("[ ");
			for (int z = 0; z < C; z++) {
				int v = val[z];
				acc += v;
				//printf("%d ", val[z]);
			}
			//printf("]\n");
		}
	}
	out[0] = acc;
}

void kernel(int in[HEIGHT * WIDTH], int out[16]) {
#pragma HLS interface axis port=in
#pragma HLS array_partition variable=in cyclic factor=16

	fifo<int_t<4,4>> ins("input_fifo");
	fifo<win_t<int_t<4,4>,3*3>> pips1("pipe_fifo1");
	fifo<int_t<4,16>> pips2("pipe_fifo2");
	fifo<int_t<4,16>> pips3("pipe_fifo3");

	Conv2D<160,160,4,3,1,2> backbone_model0_conv1;
	Conv2D<80,80,16,1> backbone_model0_conv2;

#pragma HLS dataflow
	read_input<160,160,4>(in, ins);
	backbone_model0_conv1.windowize(ins, pips1);
	backbone_model0_conv1.compute<80,80,16,7>(pips1, pips2,
		backbone_model0_conv1_weight, // [16][9]
		backbone_model0_relu1_threshold); // [16][7]
	backbone_model0_conv2.compute<80,80,16,14>(pips2, pips3,
		backbone_model0_conv2_conv1_weight, // [16][1]
		backbone_model0_conv2_quant1_threshold); // [16][14]
	write_result<80, 80, 16>(out, pips3);
}
