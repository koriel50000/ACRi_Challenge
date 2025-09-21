#include "kernel.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

const int WIDTH = 640;
const int HEIGHT = 640;
const int CHANNEL = 64;
const int FILTER = 64;

const int KERNEL = 3;
const int THRESHOLD = 14;

const int CHUNK_SIZE = 16;

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

using uint4_t = ap_uint<4>;
using uint6_t = ap_uint<6>;
using data_t = int_t<CHANNEL>;
using block_data_t = data_t[HEIGHT * WIDTH * 1];
using block_conv_t = data_t[FILTER * 1 * KERNEL * KERNEL];
using block_thr_t = int16_t[FILTER][THRESHOLD];
using win_t = hls::vector<data_t, KERNEL * KERNEL>;

template <typename T>
using fifo = hls::stream<T>;

int16_t mul(const uint4_t v, const uint4_t w) {
	static const int16_t v0[] = {
		0, 1, 2, 3, 4, 6, 8, 12,
		0, -1, -2, -3, -4, -6, -8, -12,
	};
#pragma HLS array_partition variable=v

	ap_uint<1> sign = v[3] ^ w[3];
	int16_t oval = v0[(sign, v(2, 0))] * (w(2, 0) > 0);
	return oval << (w(2, 0) - 1);
}

template <int N>
int16_t muladd(const int_t<N> vu, const int_t<N> wi) {
	static int16_t t[N];
#pragma HLS array_partition variable=t

	for (int i = 0; i < N; i++) {
#pragma HLS unroll
		t[i] = mul(vu[i], wi[i]);
	}

	for (int d = 1; d < N; d *= 2) {
		for (int i = 0; i < N; i += d * 2) {
#pragma HLS unroll
			t[i] += t[i + d];
		}
	}
	return t[0];
}

uint4_t batch_norm(const int16_t acc, const int16_t thr[], bool relu) {
	ap_uint<1> b0 = acc < thr[0];
	ap_uint<1> b1 = acc < thr[1];
	ap_uint<1> b2 = acc < thr[2];
	ap_uint<1> b3 = acc < thr[3];
	ap_uint<1> b4 = acc < thr[4];
	ap_uint<1> b5 = acc < thr[5];
	ap_uint<1> b6 = acc < thr[6];
	ap_uint<8> bits = (1, b6, b5, b4, b3, b2, b1, b0);
	// @see UG1399, Vitis HLS Coding Styles > Functions > C/C++ Builtin Functions
	return __builtin_ctz(bits);
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
public:
	void windowize(const int h, const int w, block_data_t& inb, fifo<WT>& pips) {
		LineBuffer<W + PD, KN, T, WT> linebuf(w);

		int x = 0 - (KN - 1);
		int y = 0 - (KN - 1);
		for (int i = 0; i < (W + PD) * (H + PD * 2) + PD; i++) {
#pragma HLS pipeline
			// @see UG1399, Vitis HLS Coding Styles > Loops > Variable Loop Bounds
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

	void compute(const int h, const int w, const int c,
	    block_conv_t& wi, block_thr_t& thr,
		fifo<WT>& pips, fifo<T>& outs)
	{
		for (int y = 0; y < H - (KN - 1); y++) {
			if (y >= h - (KN - 1)) break;
			for (int x = 0; x < W - (KN - 1); x++) {
				if (x >= w - (KN - 1)) break;
				WT val = pips.read();
				T oval;
				for (int j = 0; j < F; j++) {
#pragma HLS pipeline
					int16_t acc = 0;
					for (int k = 0; k < KN * KN; k++) {
					    if (c == 1) {
        					acc += mul(val[k][0], wi[j * KN * KN + k][0]);
					    } else {
        					acc += muladd<C>(val[k], wi[j * KN * KN + k]);
					    }
					}
					oval[j] = batch_norm(acc, thr[j], true);
				}
				outs.write(oval);
			}
		}
	}
};

template <int H, int W, int C>
class MaxPool2x2 {
private:
	using T = int_t<C>;

	void maxpool(const T v1, const T v2, T& ov) {
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			ov[z] = (v1[z] > v2[z]) ? v1[z] : v2[z];
		}
	}
public:
	void compute_h(const int h, const int w,
	    fifo<T>& ins, fifo<T>& pips)
	{
		for (int y = 0; y < H; y++) {
			if (y >= h) break;
			for (int x = 0; x < W; x += 2) {
#pragma HLS pipeline
				if (x >= w) break;
				T val1 = ins.read();
				T val2 = ins.read();
				T oval;
				maxpool(val1, val2, oval);
				pips.write(oval);
			}
		}
	}

	void compute_v(const int oh, const int ow,
	    block_data_t& outb, fifo<T>& pips)
	{
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
				maxpool(val1, val2, oval);
				outb[y * WIDTH + x] = oval;
			}
		}
	}
};

template <int H, int W, int C>
void read_input(fifo<uint64_t>& ins, block_data_t& outb) {
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
            ap_uint<12> v = ins.read();
			data_t val;
    		val[0] = v(11, 8);  // B(4)
    		val[1] = v(7, 4);   // G(4)
    		val[2] = v(3, 0);   // R(4)
			outb[y * WIDTH + x] = val;
		}
	}
}

void kernel(fifo<uint64_t>& in, int out[16]) {
#pragma HLS interface axis port=ins
#pragma HLS interface axis port=out

	static block_data_t even_buf;
	static block_data_t odd_buf;
	static block_conv_t even_wi;
	static block_thr_t even_thr;
	static block_conv_t odd_wi;
	static block_thr_t odd_thr;
#pragma HLS array_partition variable=even_buf cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=odd_buf cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=even_wi cyclic factor=KERNEL*KERNEL
#pragma HLS array_partition variable=even_thr
#pragma HLS array_partition variable=odd_wi cyclic factor=KERNEL*KERNEL
#pragma HLS array_partition variable=odd_thr

	read_input<640,64,3>(ins, even_buf);

//	compute_conv2d<4, 16>(buf4f, buf16b,
//		(int_t<4,4>**)backbone_model0_conv1_weight, // [16][9]
//		(int**)backbone_model0_relu1_threshold, true, // [16][7]
//		320, 320, 160, 160, 3, 1, 2);
	//compute_conv2d_1x1<16, 1>(buf16b, buf1f,
	//	(int_t<4,16>**)backbone_model0_conv2_conv1_weight, // [16][1]
	//	(int**)backbone_model0_conv2_quant1_threshold, false, // [16][14]
	//	160, 160);
	//compute_conv2d<1, 16>(buf1f, buf16b,
	//	(int_t<4,1>**)backbone_model0_conv2_conv2_weight, // [16][9]
	//	(int**)backbone_model0_conv2_relu2_threshold, true, // [16][7]
	//	160, 160, 160, 160, 3, 1);
	//compute_maxpool_2x2<16>(buf16b, buf16f,
	//	160, 160);

	//compute_conv2d_1x1<16, 1>(buf16f, buf1b,
	//	(int_t<4,16>**)backbone_model1_conv1_conv1_weight, // [16][1]
	//	(int**)backbone_model1_conv1_quant1_threshold, false, // [16][14]
	//	80, 80);
	//compute_conv2d<1, 16>(buf1b, buf16f,
	//	(int_t<4,1>**)backbone_model1_conv1_conv2_weight, // [16][9]
	//	(int**)backbone_model1_conv1_relu2_threshold, true, // [16][7]
	//	80, 80, 80, 80, 3, 1);

	//write_result<80, 80, 16>(out, buf16f);

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
