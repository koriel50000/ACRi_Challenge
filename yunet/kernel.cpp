#include "kernel.hpp"
#include "params.hpp"

void mul66(ap_uint<6> i, ap_uint<6>& o) {
	static const ap_uint<6> table[] = {
		0, 0,  0,  0,  0,  0,  0,  0,
		0, 1,  2,  3,  4,  5,  6,  7,
		0, 2,  4,  6,  8, 10, 12, 14,
		0, 3,  6,  9, 12, 15, 18, 21,
		0, 4,  8, 12, 16, 20, 24, 28,
		0, 5, 10, 15, 20, 25, 30, 35,
		0, 6, 12, 18, 24, 30, 36, 42,
		0, 7, 14, 21, 28, 35, 42, 49,
	};
	o = table[i];
}

int16_t mul(const int4_t v, const int4_t w) {
	bit_t vsign = v[3];
	bit_t wsign = w[3];
	ap_uint<3> vval = (vsign == 1) ? (-v)(2, 0) : v(2, 0);
	ap_uint<3> wval = (wsign == 1) ? (-w)(2, 0) : w(2, 0);
	ap_uint<6> oval;
	mul66((vval, wval), oval);
	return (vsign ^ wsign == 1) ? (-oval).to_int() : oval.to_int();
}

template <int C>
int16_t muladd(const int_t<4,C> vu, const int_t<4,C> wu) {
//	const int p = ilogb(C);
//	int16_t t[C];
//#pragma HLS array_partition variable=t

	int16_t acc = 0;
	for (int i = 0; i < C; i++) {
		int4_t v = vu[i];
		int4_t w = wu[i];
		acc += mul(v, w);
	}

//	for (int j = 0, d = 1; j < p; j++, d *= 2) {
//		for (int i = 0; i + d < C; i += d * 2) {
//#pragma HLS unroll
//			t[i] += t[i + d];
//		}
//	}

	return acc; //t[0];
}

int4_t batch_norm4(const int16_t acc, const int threshold[], const bool relu) {
	if (relu) {
		for (int i = 0; i < 7; i++) {
			if (acc < threshold[i]) {
				return i;
			}
		}
		return 3;
	} else {
		for (int i = 0; i < 14; i++) {
			if (acc < threshold[i]) {
				return i - 7;
			}
		}
		return 7;
	}
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

	void insert_right_col(const T value[]) {
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

template <int KN, typename T, typename WT>
class LineBuffer {
private:
	const int W;

	T buf_[MAX_SIZE * (KN - 1)];
	Window<3, 3, T, WT> window_;

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

	void get_col(T value[]) {
#pragma HLS inline
		for (int i = 0; i < KN - 1; i++) {
#pragma HLS unroll
			value[i] = buf_[i * W];
		}
	}
public:
	LineBuffer(int width) : W(width) {}

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

template <int IC, int OC>
class Conv2D {
private:
	const int H;
	const int W;
	const int KN;
	const int PD;
	const int ST;
	const int OH;
	const int OW;
public:
	Conv2D(int height, int width, int kernel, int padding, int stride, int oheight, int owidth) :
		H(height), W(width), KN(kernel), PD(padding), ST(stride), OH(oheight), OW(owidth) { }

	void windowize(fifo<int_t<4,IC>>& ins, fifo<win_t<int_t<4,IC>>>& outs) {
		LineBuffer<3, int_t<4,IC>, win_t<int_t<4,IC>>> linebuf_(W + PD);

		int x = 0 - (KN - 1);
		int y = 0 - (KN - 1);
		for (int i = 0; i < (W + PD) * (H + PD * 2) + PD; i++) {
			int_t<4,IC> val;
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
				win_t<int_t<4,IC>> oval = linebuf_.get_window();
				outs.write(oval);
			}

			x++;
			if (x >= W - (KN - 1) + PD * 2) {
				x = 0 - (KN - 1) + PD;
				y++;
			}
		}
	}

	void compute(fifo<win_t<int_t<4,IC>>>& ins, fifo<int_t<4,OC>>& outs,
		int_t<4,IC> *weight[OC], int *threshold[OC], bool relu)
	{
		for (int xy = 0; xy < OH * OW; xy++) {
			win_t<int_t<4,IC>> val = ins.read();
			int_t<4,OC> oval;
			for (int z = 0; z < OC; z++) {
				int16_t acc = 0;
				for (int k = 0; k < KN * KN; k++) {
					int_t<4,IC> v = val[k];
					int_t<4,IC> w = weight[z][k];
					acc += muladd<IC>(v, w);
				}
				//printf("%d ", acc);
				oval[z] = batch_norm4(acc, threshold[z], relu);
			}
			//printf("\n");
			outs.write(oval);
		}
	}
};

template <int C>
class MaxPool2x2 {
private:
	const int H;
	const int W;

	void maxpool(const int_t<4,C> v1, const int_t<4,C> v2, int_t<4,C>& ov) {
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			ov[z] = (v1[z] > v2[z]) ? v1[z] : v2[z];
		}
	}
public:
	MaxPool2x2(int height, int width) : H(height), W(width) {}

	void compute_h(fifo<int_t<4,C>>& ins, fifo<int_t<4,C>>& outs) {
		for (int xy = 0; xy < H * W / 2; xy++) {
#pragma HLS pipeline
			int_t<4,C> val1 = ins.read();
			int_t<4,C> val2 = ins.read();
			int_t<4,C> oval;
			maxpool(val1, val2, oval);
			outs.write(oval);
		}
	}

	void compute_v(fifo<int_t<4,C>>& ins, fifo<int_t<4,C>>& outs) {
		int_t<4,C> buf[W / 2];
#pragma HLS array_partition variable=buf

		for (int y = 0; y < H / 2; y++) {
#pragma HLS pipeline
			for (int x = 0; x < W / 2; x++) {
				buf[x] = ins.read();
			}
			for (int x = 0; x < W / 2; x++) {
				int_t<4,C> val1 = buf[x];
				int_t<4,C> val2 = ins.read();
				int_t<4,C> oval;
				maxpool(val1, val2, oval);
				outs.write(oval);
			}
		}
	}
};

template <int C>
void array_to_stream(const int size, const int_t<4,C> in[], fifo<int_t<4,C>>& ins)
{
	for (int i = 0; i < size; i++) {
#pragma HLS unroll factor=16 skip_exit_check
		int_t<4,C> val = in[i];
		ins.write(val);
	}
}

template <int C>
void stream_to_array(const int size, int_t<4,C> out[], fifo<int_t<4,C>>& outs)
{
	for (int i = 0; i < size; i++) {
#pragma HLS unroll factor=16 skip_exit_check
		int_t<4,C> val = outs.read();
		out[i] = val;
	}
}

template <int IC, int OC>
void compute_conv2d(const int_t<4,IC> in[], int_t<4,OC> out[],
	int_t<4,IC> *weight[OC], int *threshold[OC], const bool relu,
	const int height, const int width, const int oheight, const int owidth,
	const int kernel, const int padding = 0, const int stride = 1)
{
	fifo<int_t<4,IC>> ins("input_fifo");
	fifo<win_t<int_t<4,IC>>> pips1("pipe_fifo1");
	fifo<int_t<4,OC>> pips2("pipe_fifo2");
	fifo<int_t<4,OC>> outs("output_fifo");

#pragma HLS dataflow
	array_to_stream<IC>(height * width, in, ins);

	Conv2D<IC,OC> conv2d(height, width, kernel, padding, stride, oheight, owidth);
	conv2d.windowize(ins, pips1);
	conv2d.compute(pips1, pips2, weight, threshold, relu);

	stream_to_array<OC>(oheight * owidth, out, outs);
}

template <int IC, int OC>
void compute_conv2d_1x1(const int_t<4,IC> in[], int_t<4,OC> out[],
	int_t<4,IC> *weight[OC], int *threshold[OC], const bool relu,
	const int height, const int width)
{
	for (int xy = 0; xy < height * width; xy++) {
		int_t<4,IC> val = in[xy];
		int_t<4,OC> oval;
		for (int z = 0; z < OC; z++) {
			int16_t acc = muladd<IC>(val, weight[z][1]);
			oval[z] = batch_norm4(acc, threshold[z], relu);
		}
		out[xy] = oval;
	}
}

template <int C>
void compute_maxpool_2x2(const int_t<4,C> in[], int_t<4,C> out[], const int height, const int width) {
	fifo<int_t<4,C>> ins("input_fifo");
	fifo<int_t<4,C>> pips("pipe_fifo");
	fifo<int_t<4,C>> outs("output_fifo");

#pragma HLS dataflow
	int size = height * width;
	array_to_stream<C>(size, in, ins);

	MaxPool2x2<16> maxpool(height, width);
	maxpool.compute_h(ins, pips);
	maxpool.compute_v(pips, outs);

	stream_to_array<C>(size / 4, out, outs);
}

template <int H, int W, int C>
void read_input(const int in[H * W], int_t<4,C> buf[]) {
	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS unroll factor=16 skip_exit_check
		int_t<4,C> val = in[xy];
		buf[xy] = val;
	}
}

template <int H, int W, int C>
void write_result(int out[16], int_t<4,C> buf[]) {
	int acc = 0;
	for (int xy = 0; xy < H * W; xy++) {
		int_t<4,C> val = buf[xy];
		//printf("[ ");
		for (int z = 0; z < C; z++) {
			int v = val[z];
			acc += v;
			//printf("%d ", val[z]);
		}
		//printf("]\n");
	}
	out[0] = acc;
}

void kernel(int in[HEIGHT * WIDTH], int out[16]) {
#pragma HLS interface axis port=in
#pragma HLS array_partition variable=in cyclic factor=16

	int_t<4,1> buf1f[MAX_SIZE], buf1b[MAX_SIZE];
	int_t<4,4> buf4f[MAX_SIZE], buf4b[MAX_SIZE];
	int_t<4,16> buf16f[MAX_SIZE], buf16b[MAX_SIZE];
	int_t<4,64> buf64f[MAX_SIZE], buf64b[MAX_SIZE];

	read_input<320,320,4>(in, buf4f);

	compute_conv2d<4, 16>(buf4f, buf16b,
		(int_t<4,4>**)backbone_model0_conv1_weight, // [16][9]
		(int**)backbone_model0_relu1_threshold, true, // [16][7]
		320, 320, 160, 160, 3, 1, 2);
	compute_conv2d_1x1<16, 1>(buf16b, buf1f,
		(int_t<4,16>**)backbone_model0_conv2_conv1_weight, // [16][1]
		(int**)backbone_model0_conv2_quant1_threshold, false, // [16][14]
		160, 160);
	compute_conv2d<1, 16>(buf1f, buf16b,
		(int_t<4,1>**)backbone_model0_conv2_conv2_weight, // [16][9]
		(int**)backbone_model0_conv2_relu2_threshold, true, // [16][7]
		160, 160, 160, 160, 3, 1);
	compute_maxpool_2x2<16>(buf16b, buf16f,
		160, 160);

	compute_conv2d_1x1<16, 1>(buf16f, buf1b,
		(int_t<4,16>**)backbone_model1_conv1_conv1_weight, // [16][1]
		(int**)backbone_model1_conv1_quant1_threshold, false, // [16][14]
		80, 80);
	compute_conv2d<1, 16>(buf1b, buf16f,
		(int_t<4,1>**)backbone_model1_conv1_conv2_weight, // [16][9]
		(int**)backbone_model1_conv1_relu2_threshold, true, // [16][7]
		80, 80, 80, 80, 3, 1);

	write_result<80, 80, 16>(out, buf16f);

	// fifo<int_t<4,4>> ins("input_fifo");
	// fifo<win_t<int_t<4,4>,3*3>> pips1("pipe_fifo1");
	// fifo<int_t<4,16>> pips2("pipe_fifo2");
	// fifo<int_t<4,1>> pips3("pipe_fifo3");
	// fifo<win_t<int_t<4,1>,3*3>> pips4("pipe_fifo4");
	// fifo<int_t<4,16>> pips5("pipe_fifo5");
	// fifo<int_t<4,16>> pips6("pipe_fifo6");
	// fifo<int_t<4,16>> pips7("pipe_fifo7");

	// fifo<int_t<4,1>> pips8("pipe_fifo8");
	// fifo<win_t<int_t<4,1>,3*3>> pips9("pipe_fifo9");
	// fifo<int_t<4,16>> pips10("pipe_fifo10");

	// Conv2D<320,320,4,3,1,2> backbone_model0_conv1;
	// Conv2D<160,160,16,1> backbone_model0_conv2_1;
	// Conv2D<160,160,1,3,1> backbone_model0_conv2_2;
	// MaxPool2x2<160, 160, 16> backbone_model0_maxpool3;

	// Conv2D<80,80,16,1> backbone_model1_conv1_1;
	// Conv2D<80,80,1,3,1> backbone_model1_conv1_2;


	// backbone_model0_conv1.windowize(ins, pips1);
	// backbone_model0_conv1.compute<160,160,16,7>(pips1, pips2,
	// backbone_model0_conv2_1.compute<160,160,1,14>(pips2, pips3,
	// backbone_model0_conv2_2.windowize(pips3, pips4);
	// backbone_model0_conv2_2.compute<160,160,16,7>(pips4, pips5,
	// backbone_model0_maxpool3.compute_h(pips5, pips6);
	// backbone_model0_maxpool3.compute_v(pips6, pips7);

	// backbone_model1_conv1_1.compute<80,80,1,14>(pips7, pips8,
	// backbone_model1_conv1_2.windowize(pips8, pips9);
	// backbone_model1_conv1_2.compute<80,80,16,7>(pips9, pips10,
}
