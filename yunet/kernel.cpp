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

template <int H, int W, int KN, typename IT>
class Conv2D {
public:
	template <int PD = 0, int ST = 1>
	void pass_through(fifo<IT>& ins, fifo<win_t<IT,KN*KN>>& outs) {
		LineBuffer<W + PD, KN, IT, win_t<IT,KN*KN>> linebuf_;

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
				win_t<IT,KN*KN> oval = linebuf_.get_window();
				outs.write(oval);
			}
			x++;
			if (x >= W - (KN - 1) + PD * 2) {
				x = 0 - (KN - 1) + PD;
				y++;
			}
		}
	}

	template<int OH, int OW, int F, typename OT, int M>
	void compute(fifo<win_t<IT,KN*KN>>& ins, fifo<OT>& outs,
		const IT weight[F][KN*KN], const int threshold[F][M])
	{
		for (int xy = 0; xy < OH * OW; xy++) {
			win_t<IT,KN*KN> val = ins.read();
			OT oval;
			for (int z = 0; z < F; z++) {
				int16_t acc = 0;
				for (int k = 0; k < KN * KN; k++) {
					IT v = val[k];
					IT w = weight[z][k];
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
#pragma HLS unroll factor=16 skip_exit_check
		T val = in[xy];
		ins.write(val);
	}
}

template <int H, int W, int C, typename T>
void write_result(int out[16], fifo<T>& outs) {
	int acc = 0;
	for (int y = 0; y < H; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			T val = outs.read();
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
	fifo<win_t<int_t<4,16>,1*1>> pips3("pipe_fifo3");
	fifo<int_t<4,16>> pips4("pipe_fifo4");

	Conv2D<160,160,3,int_t<4,4>> backbone_model0_conv1;
	Conv2D<80,80,1,int_t<4,16>> backbone_model0_conv2;

#pragma HLS dataflow
	read_input<160, 160, int_t<4,4>>(in, ins);
	backbone_model0_conv1.pass_through<1,2>(ins, pips1);
	backbone_model0_conv1.compute<80,80,16,int_t<4,16>,7>(pips1, pips2,
		backbone_model0_conv1_weight, // [16][9]
		backbone_model0_relu1_threshold); // [16][7]
	backbone_model0_conv2.pass_through(pips2, pips3);
	backbone_model0_conv2.compute<80,80,16,int_t<4,4>,14>(pips3, pips4,
		backbone_model0_conv2_conv1_weight, // [16][1]
		backbone_model0_conv2_quant1_threshold); // [16][14]
	write_result<80, 80, 16, int_t<4,16>>(out, pips4);
}
