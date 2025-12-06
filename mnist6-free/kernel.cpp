/*
 * 4bit量子化および演算回路再利用の検証
 * ・weightを1bit符号＋3bit指数部の4bitで表現(0,1,2,4,8,16,32,64,NA,-1,-2,-4,-8,-16,-32,-64) * scale
 * ・バッチ正規化後のactivationを1bit符号+2bit指数部+1bit仮数部の4bitで表現
 *   (0,0.25,0.5,0.75,1.0,1.5,2.0,3.0, NA,-0.25,-0.5,-0.75,-1.0,-1.5,-2.0,-3.0)
 * ・推論時は閾値を4倍で計算して、activationを整数値に変換
 * ・乗算は符号なし3bitの掛け算をシフトで計算
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

template <int C>
int16_t muladd(const int c, const int_t<C> vu, const int_t<C> wi) {
	static int16_t t[C];
#pragma HLS array_partition variable=t

	for (int i = 0; i < C; i++) {
#pragma HLS unroll
		if (i >= c) break;
		t[i] = mul(vu[i], wi[i]);
	}

	for (int d = 1; d < C; d *= 2) {
		if (d >= c) break;
		for (int i = 0; i < C; i += d * 2) {
#pragma HLS unroll
			if (i >= c) break;
			t[i] += t[i + d];
		}
	}
	return t[0];
}

uint4_t batch_norm_relu(const int16_t acc, const int16_t thr[]) {
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

uint4_t batch_norm(const int16_t acc, const int16_t thr[]) {
	static const uint4_t indexTable[] = {
		7, 6, 5, 2, 4, 10, 1, 12, 0, 3, 9, 11, 15, 0, 14, 13
	};
#pragma HLS array_partition variable=indexTable

	ap_uint<1> b0 = acc < thr[0];
	ap_uint<1> b1 = acc < thr[1];
	ap_uint<1> b2 = acc < thr[2];
	ap_uint<1> b3 = acc < thr[3];
	ap_uint<1> b4 = acc < thr[4];
	ap_uint<1> b5 = acc < thr[5];
	ap_uint<1> b6 = acc < thr[6];
	ap_uint<1> b7 = acc < thr[7];
	ap_uint<1> b8 = acc < thr[8];
	ap_uint<1> b9 = acc < thr[9];
	ap_uint<1> b10 = acc < thr[10];
	ap_uint<1> b11 = acc < thr[11];
	ap_uint<1> b12 = acc < thr[12];
	ap_uint<1> b13 = acc < thr[13];
	ap_uint<15> bits = (0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13);
	// @see HD, Figure 5-26. Number of trailing zeros using a de Brujin cycle.
	// https://en.wikipedia.org/wiki/De_Bruijn_sequence
	return indexTable[((bits + 1) * 0x09af)(15, 12)];
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
	    // TODO ring buffer
		for (int i = 0; i < W * (KN - 1) - 1; i++) {
#pragma HLS pipeline
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

template <int H, int W, int C, int F, int KN>
class Conv2D {
private:
	using T = int_t<C>;
	using WT = hls::vector<T, KN * KN>;
public:
	void windowize(const int h, const int w, block_data_t& inb, fifo<WT>& pips, const int st = 1) {
		LineBuffer<W + KN - 1, KN, T, WT> linebuf(w + KN - 1);

        int x = 0 - (KN - 1) / 2;
        int y = 0 - (KN - 1) / 2;
		for (int i = 0; i < (W + KN - 1) * (H + KN - 1); i++) {
//#pragma HLS pipeline
			// @see UG1399, Vitis HLS Coding Styles > Loops > Variable Loop Bounds
			if (i >= (w + KN - 1) * (h + KN - 1)) break;
   			// input
   			T val;
    		if (0 <= x && x < w	&& 0 <= y && y < h) {
	    		val = inb[y * WIDTH + x];
		    } else {
			    val = 0;
   			}
           // buffering
   			if (i < (w + KN - 1) * (KN - 1)) {
    			linebuf.insert_linebuf(val);
	    	} else {
			    linebuf.slide_window(val);
   			}
 			// output
   			if (0 + (KN - 1) / 2 <= x && 0 + (KN - 1) / 2 <= y
   			    && (x - (KN - 1) / 2) % st == 0 && (y - (KN - 1) / 2) % st == 0)
   			{
    			WT oval = linebuf.get_window();
	    		pips.write(oval);
	    	}
		    x++;
		    if (x >= w + (KN - 1) / 2) {
                x = 0 - (KN - 1) / 2;
		        y++;
		    }
		}
	}

	void compute(const int h, const int w, const int c, const int f, const bool relu,
		block_conv_t& wi, block_thr_t& thr,
		fifo<WT>& pips, block_data_t& outb)
	{
		for (int y = 0; y < H; y++) {
			if (y >= h) break;
			for (int x = 0; x < W; x++) {
				if (x >= w) break;
				WT val = pips.read();
				T oval;
				for (int j = 0; j < F; j++) {
//#pragma HLS pipeline
					if (j >= f) break;
					int16_t acc = 0;
					for (int k = 0; k < KN * KN; k++) {
						acc += muladd<C>(c, val[k], wi[j * KN * KN + k]);
					}
					if (relu) {
    					oval[j] = batch_norm_relu(acc, thr[j]);
	   				} else {
		    			oval[j] = batch_norm(acc, thr[j]);
					}
				}
				outb[y * WIDTH + x] = oval;
			}
		}
	}
};

template <int H, int W, int C, int F>
class Conv2D1x1 {
private:
	using T = int_t<C>;
public:
	void compute(const int h, const int w, const int c, const int f,
		block_conv_t& wi, block_thr_t& thr,
		block_data_t& inb, block_data_t& outb)
	{
		for (int y = 0; y < H; y++) {
			if (y >= h) break;
			for (int x = 0; x < W; x++) {
				if (x >= w) break;
				T val = inb[y * WIDTH + x];
				T oval;
				for (int j = 0; j < F; j++) {
//#pragma HLS pipeline
					if (j >= f) break;
					int16_t acc = muladd<C>(c, val, wi[j]);
					oval[j] = batch_norm(acc, thr[j]);
				}
				outb[y * WIDTH + x] = oval;
			}
		}
	}
};

template <int H, int W, int C>
class MaxPool2x2 {
private:
	using T = int_t<C>;

	void maxpool(const int c, const T v1, const T v2, T& ov) {
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			if (z >= c) break;
			ov[z] = (v1[z] > v2[z]) ? v1[z] : v2[z];
		}
	}
public:
	void compute_h(const int h, const int w, const int c,
		block_data_t& inb, fifo<T>& pips)
	{
		for (int y = 0; y < H; y++) {
			if (y >= h) break;
			for (int x = 0; x < W; x += 2) {
#pragma HLS pipeline
				if (x >= w) break;
				T val1 = inb[y * WIDTH + x];
				T val2 = inb[y * WIDTH + x + 1];
				T oval;
				maxpool(c, val1, val2, oval);
				pips.write(oval);
			}
		}
	}

	void compute_v(const int oh, const int ow, const int oc,
		fifo<T>& pips, block_data_t& outb)
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
				maxpool(oc, val1, val2, oval);
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
#pragma HLS pipeline
			if (acc[i] > max) {
				max = acc[i];
				m = i;
			}
		}
		out[0] = m;
	}
};

void read_data(const int h, const int w, const int c,
	fifo<uint64_t>& ins, block_data_t& outb)
{
	for (int y = 0; y < HEIGHT; y++) {
		if (y >= h) break;
		for (int x = 0; x < WIDTH; x++) {
			if (x >= w) break;
			data_t val = data_t(ins.read());
			outb[y * WIDTH + x] = val;
		}
	}
}

void read_weight(const int f, const int c, const int kn,
	fifo<uint64_t>& ins, block_conv_t& outw, block_thr_t& outh)
{
	for (int i = 0; i < FILTER * KERNEL * KERNEL; i++) {
		if (i >= f * kn * kn) break;
		outw[i] = data_t(ins.read());
	}

	for (int j = 0; j < FILTER; j++) {
		if (j >= f) break;
		for (int i = 0; i < THRESHOLD; i++) {
			outh[j][i] = ins.read();
		}
	}
}

Conv2D<HEIGHT,WIDTH,CHANNEL,FILTER,3> conv3x3;
Conv2D1x1<HEIGHT,WIDTH,CHANNEL,FILTER> conv1x1;
MaxPool2x2<HEIGHT,WIDTH,CHANNEL> maxpool2x2;

void read_compute_conv3x3_stride(const int h, const int w, const int c, const int f,
    const int nf, const int nc, const int nkn,
    fifo<uint64_t>& ins,
	block_conv_t& cur_wi, block_thr_t& cur_thr,
	block_conv_t& next_wi, block_thr_t& next_thr,
	block_data_t& inb, block_data_t& outb)
{
	fifo<win_t> pips1("pipe_fifo1");

#pragma HLS dataflow
	read_weight(nf, nc, nkn, ins, next_wi, next_thr);
	conv3x3.windowize(h, w, inb, pips1, 2);
	conv3x3.compute(h / 2, w / 2, c, f, true, cur_wi, cur_thr, pips1, outb);
}

void read_compute_conv3x3_relu(const int h, const int w, const int c, const int f,
    const int nf, const int nc, const int nkn,
    fifo<uint64_t>& ins,
	block_conv_t& cur_wi, block_thr_t& cur_thr,
	block_conv_t& next_wi, block_thr_t& next_thr,
	block_data_t& inb, block_data_t& outb)
{
	fifo<win_t> pips1("pipe_fifo1");

#pragma HLS dataflow
	read_weight(nf, nc, nkn, ins, next_wi, next_thr);
	conv3x3.windowize(h, w, inb, pips1);
	conv3x3.compute(h, w, c, f, true, cur_wi, cur_thr, pips1, outb);
}

void read_compute_conv1x1(const int h, const int w, const int c, const int f,
    const int nf, const int nc, const int nkn,
    fifo<uint64_t>& ins,
	block_conv_t& cur_wi, block_thr_t& cur_thr,
	block_conv_t& next_wi, block_thr_t& next_thr,
	block_data_t& inb, block_data_t& outb)
{
#pragma HLS dataflow
	read_weight(nf, nc, nkn, ins, next_wi, next_thr);
	conv1x1.compute(h, w, c, f, cur_wi, cur_thr, inb, outb);
}

void compute_maxpool2x2(const int h, const int w, const int c,
	block_data_t& inb, block_data_t& outb)
{
	fifo<data_t> pips1("pipe_fifo1");

#pragma HLS dataflow
	maxpool2x2.compute_h(h, w, c, inb, pips1);
	maxpool2x2.compute_v(h / 2, w / 2, c, pips1, outb);
}

void print_data_hist(const int h, const int w, const int c, block_data_t& buf) {
    int count = 0;
    float sum = 0;
    int hist[15] = {};
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            data_t val = buf[y * WIDTH + x];
            for (int z = 0; z < c; z++) {
                int v = val[z].to_int();
                count++;
                sum += v;
                hist[v]++;
                if (count <= 20) {
                    printf("%d ", v);
                }
            }
        }
    }
    printf("\n");
    printf("mean=%f\n", sum / count);
    for (int i = 15; i > 8; --i) {
        printf("[%d]=%d ", 8 - i, hist[i]);
    }
    for (int i = 0; i < 8; i++) {
        printf("[%d]=%d ", i, hist[i]);
    }
    printf("\n");
}

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

	read_data(160, 160, 3, ins, even_buf);
	read_weight(16, 3, 3, ins, even_wi, even_thr);
	// Conv_head
	read_compute_conv3x3_stride(160, 160, 3, 16, 16, 16, 1,
	    ins, even_wi, even_thr, odd_wi, odd_thr, even_buf, odd_buf);
	// Conv_head ConvDPUnit
	read_compute_conv1x1(80, 80, 16, 16, 16, 1, 3,
	    ins, odd_wi, odd_thr, even_wi, even_thr, odd_buf, even_buf);
	read_compute_conv3x3_relu(80, 80, 16, 1, 16, 16, 1,
	    ins, even_wi, even_thr, odd_wi, odd_thr, even_buf, odd_buf);
	// YuNetBackbone
	compute_maxpool2x2(80, 80, 16, odd_buf, even_buf);
	write_compute(out, mat_wi, even_buf);
}
