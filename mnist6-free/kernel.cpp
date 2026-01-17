/*
 * YuNet顔検出の4bit量子化および演算回路再利用に向けた事前検証
 * ・weightを1bit符号＋3bit指数部の4bitで表現(0,1,2,4,8,16,32,64,NA,-1,-2,-4,-8,-16,-32,-64) * scale
 * ・バッチ正規化を追加し、activationを1bit符号+2bit指数部+1bit仮数部の4bitで表現
 *   (0,0.25,0.5,0.75,1.0,1.5,2.0,3.0, NA,-0.25,-0.5,-0.75,-1.0,-1.5,-2.0,-3.0)
 * ・推論時は閾値を4倍で計算して、activationを整数値に変換
 * ・乗算は符号なし3bitの掛け算をシフトで計算
 * ・演算回路は最大サイズのConv,Maxpoolを用意し、引数で行列サイズを指定して再利用(ループをbreak、閾値の範囲外は0埋め)
 * ・メインメモリから入力画像、重みおよび閾値のパラメータを転送し続ける
 * ・ダブルバッファリングで、パラメータ転送中に演算して演算結果を一時保存
 */
/*
-- CNV.py --
class CNV(nn.Module):

    def __init__(self):
        super(CNV, self).__init__()

        self.conv1 = qnn.QuantConv2d(1, 16, 3, 2, 1, bias=False,
                                     weight_quant=Int4WeightQuant, weight_bit_width=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = qnn.QuantReLU(act_quant=Int4ActQuant)
        self.conv2 = qnn.QuantConv2d(16, 16, 1, 1, 0, bias=False,
                                     weight_quant=Int4WeightQuant, weight_bit_width=4)
        self.bn2 = nn.BatchNorm2d(16)
        self.quant2 = qnn.QuantIdentity(act_quant=Int4ActQuant)
        self.conv3 = qnn.QuantConv2d(16, 16, 3, 1, 1, bias=False,
                                     weight_quant=Int4WeightQuant, weight_bit_width=4)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = qnn.QuantReLU(act_quant=Int4ActQuant)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.linear = qnn.QuantLinear(784, 10, bias=False,
                                      weight_quant=Int4WeightQuant, weight_bit_width=4)
        self.tn = TensorNorm()


-- common.py --
class Fp4e3m0Mixin(ExtendedInjector):
    bit_width = 4
    exponent_bit_width = 3
    mantissa_bit_width = 0
    saturating = True


class Int4WeightQuant(Fp4e3m0Mixin,
                      ScaledFloatWeightBase):
    scaling_per_output_type = ScalingPerOutputType.CHANNEL

    @value
    def exponent_bias(exponent_bit_width):
        return 1


class Int4ActQuant(Fp4e2m1Mixin,
                   FloatActBase,
                   ActQuantSolver):
    scaling_impl_type = ScalingImplType.CONST
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    scaling_const = 1
    max_val = 0.5
    min_val = -0.5
 */
#include "kernel.hpp"
#include "layers.hpp"

void print_data_hist(const int h, const int w, const int c, block_data_t& buf) {
    int count = 0;
    float sum = 0;
    int hist[16] = {};
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

void read_input(const int h, const int w, const int c,
	fifo<uint64_t>& ins, block_data_t& outb)
{
	for (int y = 0; y < HEIGHT; y++) {
		if (y >= h) break;
		for (int x = 0; x < WIDTH; x++) {
			if (x >= w) break;
			data_t val = data_t(12 - ins.read() * 8);
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

void read_mat_weight(fifo<uint64_t>& ins, block_mat_t& mat_wi) {
	for (int i = 0; i < CLASS * FLATTEN / CHUNK_SIZE; i++) {
	    mat_wi[i] = data_t(ins.read());
	}
}

Conv2D<HEIGHT,WIDTH,CHANNEL,FILTER,KERNEL> conv3x3;
Conv2D1x1<HEIGHT,WIDTH,CHANNEL,FILTER> conv1x1;
MaxPool2x2<HEIGHT,WIDTH,CHANNEL> maxpool2x2;
Dense<CLASS,FLATTEN,CHUNK_SIZE,7,7> matmul;

void read_compute_conv3x3_stride(
    const int h, const int w, const int c, const int f,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& inb, block_data_t& outb,
    const int nf, const int nc, const int nkn,
    fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
	fifo<win_t> pips1("pipe_fifo1");

#pragma HLS dataflow
	conv3x3.windowize(h, w, inb, pips1, 2);
	conv3x3.compute(h / 2, w / 2, c, f, true, cur_wi, cur_thr, pips1, outb);
	read_weight(nf, nc, nkn, ins, next_wi, next_thr);
}

void read_compute_conv1x1(
    const int h, const int w, const int c, const int f,
	block_conv_t& cur_wi, block_thr_t& cur_thr,
	block_data_t& inb, block_data_t& outb,
    const int nf, const int nc, const int nkn,
    fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
#pragma HLS dataflow
	conv1x1.compute(h, w, c, f, cur_wi, cur_thr, inb, outb);
	read_weight(nf, nc, nkn, ins, next_wi, next_thr);
}

void read_compute_conv3x3_relu(
    const int h, const int w, const int c, const int f,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& inb, block_data_t& outb,
    fifo<uint64_t>& ins, block_mat_t& mat_wi)
{
	fifo<win_t> pips1("pipe_fifo1");

#pragma HLS dataflow
	conv3x3.windowize(h, w, inb, pips1);
	conv3x3.compute(h, w, c, f, true, cur_wi, cur_thr, pips1, outb);
	read_mat_weight(ins, mat_wi);
}

void compute_maxpool2x2(const int h, const int w, const int c,
	block_data_t& inb, block_data_t& outb)
{
	fifo<data_t> pips1("pipe_fifo1");

#pragma HLS dataflow
	maxpool2x2.compute_h(h, w, c, inb, pips1);
	maxpool2x2.compute_v(h / 2, w / 2, c, pips1, outb);
}

void compute_matmul_write(block_mat_t& mat_wi, block_data_t& inb, int out[1]) {
    fifo<int_t<CLASS,16>> pips("pipe_fifo");  // int16_t

#pragma HLS dataflow
	matmul.flatten(mat_wi, inb, pips);
	matmul.write_result(out, pips);
}

void kernel_inner(fifo<uint64_t>& ins, int out[1]) {
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
#pragma HLS array_partition variable=mat_wi cyclic factor=CHUNK_SIZE

const int W = 28;
const int H = 28;
const int KN = 3;
int w = 10;
int h = 10;
using win = hls::vector<int, KN * KN>;
LineBuffer<W + (KN - 1), KN, int, win> linebuf(w + (KN - 1));
int x = 0 - (KN - 1) / 2;
int y = 0 - (KN - 1) / 2;
for (int i = 0; i < (W + (KN - 1)) * (H + (KN - 1)); i++) {
  if (i >= (w + KN - 1) * (h + KN - 1)) break;
  // input
  int val;
  if (0 <= x && x < w && 0 <= y && y < h) {
    val = y * w + x + 1;
  } else {
    val = 0;
  }
  // buffering
  if (i < (w + (KN - 1)) * (KN - 1)) {
    linebuf.insert_linebuf(val);
  } else {
    linebuf.slide_window(val);
  }
  // output
  if (1 <= x && 1 <= y)
  {
    win oval = linebuf.get_window();
    for (int ky = 0; ky < KN; ky++) {
      for (int kx = 0; kx < KN; kx++) {
        printf("[%d,%d]=%d ", x - (KN - 1) + kx, y - (KN - 1) + ky, oval[ky * KN + kx]);
      }
      printf("\n");
	}
    printf("\n");
  }
  x++;
  if (x >= w + (KN - 1) / 2) {
    x = 0 - (KN - 1) / 2;
    y++;
  }
}
return;
	read_input(28, 28, 1, ins, even_buf);
	read_weight(16, 3, 3, ins, even_wi, even_thr);
	// Conv_head
	read_compute_conv3x3_stride(
	    14, 14, 1, 16, even_wi, even_thr, even_buf, odd_buf,
	    16, 1, 1, ins, odd_wi, odd_thr);
	// Conv_head ConvDPUnit
	read_compute_conv1x1(
	    14, 14, 16, 16, odd_wi, odd_thr, odd_buf, even_buf,
	    16, 3, 3, ins, even_wi, even_thr);
	read_compute_conv3x3_relu(
		14, 14, 16, 16, even_wi, even_thr, even_buf, odd_buf,
	    ins, mat_wi);
	// YuNetBackbone
	compute_maxpool2x2(14, 14, 16, odd_buf, even_buf);
	compute_matmul_write(mat_wi, even_buf, out);
}
