/*
 * 4bit量子化および演算回路再利用の検証
 * ・weightを1bit符号＋3bit指数部の4bitで表現(0,1,2,4,8,16,32,64,NA,-1,-2,-4,-8,-16,-32,-64) * scale
 * ・バッチ正規化後のactivationを1bit符号+2bit指数部+1bit仮数部の4bitで表現
 *   (0,0.25,0.5,0.75,1.0,1.5,2.0,3.0, NA,-0.25,-0.5,-0.75,-1.0,-1.5,-2.0,-3.0)
 * ・乗算は符号なし3bitの掛け算を、6入力LUTが4個のテーブル参照とシフトで計算
 * ・演算回路は最大サイズのConv,Maxpoolを用意し、引数で行列サイズを指定して再利用(ループをbreak?範囲外は0埋め?)
 * ・conv0_wi,conv0_thr -> in -> conv1_wi,conv1_thr -> mat_wi の順にメインメモリからパラメータを転送
 * ・ダブルバッファリングで、パラメータ転送中に演算して演算結果を一時保存
 */
/*
-- CNV.py --
class CNV(nn.Module):

	def __init__(self, num_classes=10, weight_bit_width=4, act_bit_width=4, in_bit_width=1, in_channels=1):
		super(CNV, self).__init__()

		self.conv_features = nn.ModuleList()
		self.linear_features = nn.ModuleList()

		self.conv_features.append(qnn.QuantConv2d(
			kernel_size=KERNEL_SIZE,
			in_channels=in_channels,
			out_channels=16,
			bias=False,
			weight_quant=CommonWeightQuant,
			weight_bit_width=weight_bit_width))
		self.conv_features.append(nn.BatchNorm2d(16, eps=1e-4))
		self.conv_features.append(qnn.QuantReLU(
			act_quant=CommonActQuant,
			bit_width=act_bit_width,
			return_quant_tensor=True))
		self.conv_features.append(nn.MaxPool2d(kernel_size=POOL_SIZE))

		self.conv_features.append(qnn.QuantConv2d(
			kernel_size=KERNEL_SIZE,
			in_channels=16,
			out_channels=16,
			bias=False,
			weight_quant=CommonWeightQuant,
			weight_bit_width=weight_bit_width))
		self.conv_features.append(nn.BatchNorm2d(16, eps=1e-4))
		self.conv_features.append(qnn.QuantReLU(
			act_quant=CommonActQuant,
			bit_width=act_bit_width,
			return_quant_tensor=True))
		self.conv_features.append(nn.MaxPool2d(kernel_size=POOL_SIZE))

		self.linear_features.append(qnn.QuantLinear(
			in_features=256,
			out_features=num_classes,
			bias=False,
			weight_quant=CommonWeightQuant,
			weight_bit_width=weight_bit_width))
		self.linear_features.append(TensorNorm())


-- common.py --
class Fp4e3m0Mixin(ExtendedInjector):
	bit_width = 4
	exponent_bit_width = 3
	mantissa_bit_width = 0
	saturating = True


class CommonWeightQuant(Fp4e3m0Mixin,
			ScaledFloatWeightBase):
	scaling_per_output_type = ScalingPerOutputType.TENSOR

	@value
	def exponent_bias(exponent_bit_width):
		return 1


class CommonActQuant(Fp4e2m1Mixin,
			 FloatActBase,
			 ActQuantSolver):
	scaling_impl_type = ScalingImplType.CONST
	scaling_per_output_channel = False
	restrict_scaling_type = RestrictValueType.FP
	scaling_const = 1.0
	max_val = 0.5
	min_val = -0.5
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
using block_data_t = data_t[HEIGHT * WIDTH];
using block_conv_t = data_t[FILTER * KERNEL * KERNEL];
using block_thr_t = int16_t[FILTER][THRESHOLD];
using block_mat_t = data_t[CLASS * FLATTEN / CHUNK_SIZE];
using win_t = hls::vector<data_t, KERNEL * KERNEL>;

template <typename T>
using fifo = hls::stream<T>;
template <typename T>
using sob = hls::stream_of_blocks<T>;

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
	ap_uint<1> sign = v[3] ^ w[3];
	int16_t oval = mul64((v(2, 0), w(2, 0)));
	oval = oval << ((w(1, 0) + 1) & -w[2]);
	return (oval ^ -sign) + sign;
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

template <int CL, int FL, int K, int H, int W>
class Dense {
private:
	using IT = int_t<K>;
	using OT = int_t<CL,16>;
public:
	void flatten(block_mat_t& mat, block_data_t& inb, fifo<OT>& pips) {
		int ptr = 0;
		for (int y = 0; y < H; y++) {
			for (int x = 0; x < W; x++) {
				IT vu = inb[y * WIDTH + x];
				OT oval;
				for (int i = 0; i < CL; i++) {
#pragma HLS pipeline
					IT wi = mat[ptr++];
					int16_t acc = muladd<K>(vu, wi);
					oval[i] = acc;
				}
				pips.write(oval);
			}
		}
	}

	void write_result(int out[1], fifo<OT>& pips) {
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

		int16_t max = -10000;
		int m = 0;
		for (int i = 0; i < CL; i++) {
#pragma HLS unroll
			if (acc[i] > max) {
				max = acc[i];
				m = i;
			}
		}
		out[0] = m;
	}
};

template <int H, int W, int C>
void read_input(fifo<long>& ins, block_data_t& outb) {
	int ptr = 0;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
			data_t val;
			for (int z = 0; z < C; z++) {
#pragma HLS unroll
				val[z] = ins.read() * 8 - 4;
			}
			outb[y * WIDTH + x] = val;
		}
	}
}

void read_weight1(fifo<long>& ins, block_conv_t& conv_wi, block_thr_t& conv_thr) {
	for (int i = 0; i < FILTER * KERNEL * KERNEL; i++) {
	    conv_wi[i] = data_t(ins.read());
	}

	for (int j = 0; j < FILTER; j++) {
    	for (int i = 0; i < THRESHOLD; i++) {
	        conv_thr[j][i] = ins.read();
    	}
	}
}

void read_weight2(fifo<long>& ins, block_conv_t& conv_wi, block_thr_t& conv_thr) {
	for (int i = 0; i < FILTER * KERNEL * KERNEL; i++) {
	    conv_wi[i] = data_t(ins.read());
	}

	for (int j = 0; j < FILTER; j++) {
    	for (int i = 0; i < THRESHOLD; i++) {
	        conv_thr[j][i] = ins.read();
    	}
	}
}

void read_weight3(fifo<long>& ins, block_mat_t& mat_wi) {
	for (int i = 0; i < CLASS * FLATTEN / CHUNK_SIZE; i++) {
	    mat_wi[i] = data_t(ins.read());
	}
}

Conv2D<HEIGHT,WIDTH,CHANNEL,FILTER,KERNEL> conv;
MaxPool2x2<HEIGHT,WIDTH,CHANNEL> maxpool;
Dense<CLASS,FLATTEN,CHUNK_SIZE,4,4> matmul0;

void read_compute1(fifo<long>& ins, block_conv_t& cur_wi, block_thr_t& cur_thr,
    block_conv_t& next_wi, block_thr_t& next_thr,
    block_data_t& inb, block_data_t& outb)
{
	fifo<win_t> pips1("pipe_fifo1");
	fifo<data_t> pips2("pipe_fifo2");
	fifo<data_t> pips3("pipe_fifo3");

#pragma HLS dataflow
    read_weight2(ins, next_wi, next_thr);
	conv.windowize(28, 28, inb, pips1);
	conv.compute(28, 28, 1, cur_wi, cur_thr, pips1, pips2);
	maxpool.compute_h(24, 24, pips2, pips3);
	maxpool.compute_v(12, 12, outb, pips3);
}

void read_compute2(fifo<long>& ins, block_conv_t& cur_wi, block_thr_t& cur_thr, block_mat_t& mat_wi,
    block_data_t& inb, block_data_t& outb)
{
	fifo<win_t> pips1("pipe_fifo1");
	fifo<data_t> pips2("pipe_fifo2");
	fifo<data_t> pips3("pipe_fifo3");

#pragma HLS dataflow
    read_weight3(ins, mat_wi);
	conv.windowize(12, 12, inb, pips1);
	conv.compute(12, 12, 16, cur_wi, cur_thr, pips1, pips2);
	maxpool.compute_h(8, 8, pips2, pips3);
	maxpool.compute_v(4, 4, outb, pips3);
}

void write_compute3(int out[1], block_mat_t& mat_wi, block_data_t& inb) {
    fifo<int_t<CLASS,16>> pips("pipe_fifo");

#pragma HLS dataflow
	matmul0.flatten(mat_wi, inb, pips);
	matmul0.write_result(out, pips);
}

void kernel(fifo<long>& ins, int out[1]) {
#pragma HLS interface axis port=ins
#pragma HLS interface axis port=out

	static block_data_t even_buf;
	static block_data_t odd_buf;
	static block_conv_t even_wi;
	static block_thr_t even_thr;
	static block_conv_t odd_wi;
	static block_thr_t odd_thr;
	static block_mat_t mat_wi;
#pragma HLS array_partition variable=even_buf cyclic factor=WIDTH
#pragma HLS array_partition variable=odd_buf cyclic factor=WIDTH
#pragma HLS array_partition variable=even_wi cyclic factor=KERNEL*KERNEL
#pragma HLS array_partition variable=even_thr
#pragma HLS array_partition variable=odd_wi cyclic factor=KERNEL*KERNEL
#pragma HLS array_partition variable=odd_thr
#pragma HLS array_partition variable=mat_wi cyclic factor=FLATTEN/CHUNK_SIZE

	read_input<28,28,1>(ins, even_buf);
    read_weight1(ins, even_wi, even_thr);
	read_compute1(ins, even_wi, even_thr, odd_wi, odd_thr, even_buf, odd_buf);
	read_compute2(ins, odd_wi, odd_thr, mat_wi, odd_buf, even_buf);
	write_compute3(out, mat_wi, even_buf);
}
