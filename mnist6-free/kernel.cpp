/*
やっとBrevitasで学習したパラメータからFPGAで推論を動かすことができました。
https://github.com/Xilinx/brevitas/tree/master/src/brevitas_examples/bnn_pynq

＃HLSチャレンジで閾値(threshold)が1組なのはナゾ ;-)

-- CNV.py --
class CNV(nn.Module):

	def __init__(self, num_classes=10, weight_bit_width=2, act_bit_width=2, in_bit_width=1, in_channels=1):
		super(CNV, self).__init__()

		self.conv_features = nn.ModuleList()
		self.linear_features = nn.ModuleList()

		self.conv_features.append(bnn.QuantIdentity(act_quant=CommonInQuant, bit_width=in_bit_width))

		self.conv_features.append(
			bnn.QuantConv2d(
			kernel_size=KERNEL_SIZE,
			in_channels=in_channels,
			out_channels=16,
			bias=False,
			weight_quant=CommonWeightQuant,
			weight_bit_width=weight_bit_width))
		self.conv_features.append(nn.BatchNorm2d(16, eps=1e-4))
		self.conv_features.append(bnn.QuantReLU(
			act_quant=CommonActQuant,
			bit_width=act_bit_width + 1,
			return_quant_tensor=True))
		self.conv_features.append(nn.MaxPool2d(kernel_size=POOL_SIZE))

		self.conv_features.append(
			bnn.QuantConv2d(
			kernel_size=KERNEL_SIZE,
			in_channels=16,
			out_channels=16,
			bias=False,
			weight_quant=CommonWeightQuant,
			weight_bit_width=weight_bit_width))
		self.conv_features.append(nn.BatchNorm2d(16, eps=1e-4))
		self.conv_features.append(bnn.QuantReLU(
			act_quant=CommonActQuant,
			bit_width=act_bit_width + 1,
			return_quant_tensor=True))
		self.conv_features.append(nn.MaxPool2d(kernel_size=POOL_SIZE))

		self.linear_features.append(
			bnn.QuantLinear(
			in_features=256,
			out_features=num_classes,
			bias=False,
			weight_quant=CommonWeightQuant,
			weight_bit_width=weight_bit_width))
		self.linear_features.append(TensorNorm())

		for mod in self.modules():
			if isinstance(mod, bnn.QuantConv2d) or isinstance(mod, bnn.QuantLinear):
				nn.init.uniform_(mod.weight.data, -1, 1)

	def clip_weights(self, min_val, max_val):
		for mod in self.modules():
			if isinstance(mod, bnn.QuantConv2d) or isinstance(mod, bnn.QuantLinear):
				mod.weight.data.clamp_(min_val, max_val)


-- common.py --
class CommonActQuant(CommonQuant, ActQuantSolver):
	min_val = -3.0
	max_val = 3.0


class CommonInQuant(CommonQuant, ActQuantSolver):
	min_val = -1.0
	max_val = 1.0

 */
#include "kernel.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#define I16(i) int_t<1,16>({ (i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1, (i >> 4) & 1, (i >> 5) & 1, (i >> 6) & 1, (i >> 7) & 1, (i >> 8) & 1, (i >> 9) & 1, (i >> 10) & 1, (i >> 11) & 1, (i >> 12) & 1, (i >> 13) & 1, (i >> 14) & 1, (i >> 15) & 1, })
#define I25(i) int_t<1,25>({ (i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1, (i >> 4) & 1, (i >> 5) & 1, (i >> 6) & 1, (i >> 7) & 1, (i >> 8) & 1, (i >> 9) & 1, (i >> 10) & 1, (i >> 11) & 1, (i >> 12) & 1, (i >> 13) & 1, (i >> 14) & 1, (i >> 15) & 1, (i >> 16) & 1, (i >> 17) & 1, (i >> 18) & 1, (i >> 19) & 1, (i >> 20) & 1, (i >> 21) & 1, (i >> 22) & 1, (i >> 23) & 1, (i >> 24) & 1, })

const int WIDTH = 28;
const int HEIGHT = 28;

const int KERNEL = 5;

const ap_uint<1> b0w1 = 0;

template <int W, int N>
using int_t = hls::vector<ap_uint<W>, N>;
template <typename T>
using fifo = hls::stream<T>;

using bit_t = ap_uint<1>;
using uint2_t = ap_uint<2>;
using uint3_t = ap_uint<3>;
using int3_t = ap_int<3>;
using uint4_t = ap_uint<4>;
using int4_t = ap_int<4>;
using uint6_t = ap_uint<6>;
using wpack_t = hls::vector<int_t<2,16>, KERNEL * KERNEL>;

void mac63(uint6_t i, int3_t& o) {
	static const int3_t table[] = {
		0, 1, 1, 2, -1, 0, 0, 1,
		-1, 0, 0, 1, -2, -1, -1, 0,
		0, -1, 1, 0, 1, 0, 2, 1,
		-1, -2, 0, -1, 0, -1, 1, 0,
		0, 1, -1, 0, -1, 0, -2, -1,
		1, 2, 0, 1, 0, 1, -1, 0,
		0, -1, -1, -2, 1, 0, 0, -1,
		1, 0, 0, -1, 2, 1, 1, 0,
	};
	o = table[i];
}

void ac64(uint6_t i, int4_t& o) {
	static const int4_t table[] = {
		0, 1, 2, 3, -4, -3, -2, -1,
		1, 2, 3, 4, -3, -2, -1, 0,
		2, 3, 4, 5, -2, -1, 0, 1,
		3, 4, 5, 6, -1, 0, 1, 2,
		-4, -3, -2, -1, -8, -7, -6, -5,
		-3, -2, -1, 0, -7, -6, -5, -4,
		-2, -1, 0, 1, -6, -5, -4, -3,
		-1, 0, 1, 2, -5, -4, -3, -2,
	};
	o = table[i];
}

int16_t muladd25(int_t<1,25> vu, int_t<1,25> wp, int_t<1,25> wn) {
	int3_t c0100, c0302, c0504, c0706, c0908, c1110, c1312, c1514;
	int3_t c1716, c1918, c2120, c2322, c24;

	mac63((vu[ 1], vu[ 0], wp[ 1], wp[ 0], wn[ 1], wn[ 0]), c0100);
	mac63((vu[ 3], vu[ 2], wp[ 3], wp[ 2], wn[ 3], wn[ 2]), c0302);
	mac63((vu[ 5], vu[ 4], wp[ 5], wp[ 4], wn[ 5], wn[ 4]), c0504);
	mac63((vu[ 7], vu[ 6], wp[ 7], wp[ 6], wn[ 7], wn[ 6]), c0706);
	mac63((vu[ 9], vu[ 8], wp[ 9], wp[ 8], wn[ 9], wn[ 8]), c0908);
	mac63((vu[11], vu[10], wp[11], wp[10], wn[11], wn[10]), c1110);
	mac63((vu[13], vu[12], wp[13], wp[12], wn[13], wn[12]), c1312);
	mac63((vu[15], vu[14], wp[15], wp[14], wn[15], wn[14]), c1514);
	mac63((vu[17], vu[16], wp[17], wp[16], wn[17], wn[16]), c1716);
	mac63((vu[19], vu[18], wp[19], wp[18], wn[19], wn[18]), c1918);
	mac63((vu[21], vu[20], wp[21], wp[20], wn[21], wn[20]), c2120);
	mac63((vu[23], vu[22], wp[23], wp[22], wn[23], wn[22]), c2322);
	mac63((b0w1, vu[24], b0w1, wp[24], b0w1, wn[24]), c24);

	int4_t c0, c1, c2, c3, c4, c5;

	ac64((c0100, c0302), c0);
	ac64((c0504, c0706), c1);
	ac64((c0908, c1110), c2);
	ac64((c1312, c1514), c3);
	ac64((c1716, c1918), c4);
	ac64((c2120, c2322), c5);

	return ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + c24);
}

void muac63(uint6_t i, uint3_t& o) {
	static const uint3_t table[] = {
		0, 0, 0, 0, 0, 1, 0, 1,
		0, 2, 0, 2, 0, 3, 0, 3,
		0, 0, 1, 1, 0, 1, 1, 2,
		0, 2, 1, 3, 0, 3, 1, 4,
		0, 0, 2, 2, 0, 1, 2, 3,
		0, 2, 2, 4, 0, 3, 2, 5,
		0, 0, 3, 3, 0, 1, 3, 4,
		0, 2, 3, 5, 0, 3, 3, 6,
	};
	o = table[i];
}

void uac64(uint6_t i, uint4_t& o) {
	static const uint4_t table[] = {
		0, 1, 2, 3, 4, 5, 6, 7,
		1, 2, 3, 4, 5, 6, 7, 8,
		2, 3, 4, 5, 6, 7, 8, 9,
		3, 4, 5, 6, 7, 8, 9, 10,
		4, 5, 6, 7, 8, 9, 10, 11,
		5, 6, 7, 8, 9, 10, 11, 12,
		6, 7, 8, 9, 10, 11, 12, 13,
		7, 8, 9, 10, 11, 12, 13, 14,
	};
	o = table[i];
}

int16_t muluadd32(int_t<2,16> vu, int_t<1,16> wp, int_t<1,16> wn) {
	uint3_t p0100, p0302, p0504, p0706, p0908, p1110, p1312, p1514;
	uint3_t n0100, n0302, n0504, n0706, n0908, n1110, n1312, n1514;

	muac63((vu[ 1], vu[ 0], wp[ 1], wp[ 0]), p0100);
	muac63((vu[ 3], vu[ 2], wp[ 3], wp[ 2]), p0302);
	muac63((vu[ 5], vu[ 4], wp[ 5], wp[ 4]), p0504);
	muac63((vu[ 7], vu[ 6], wp[ 7], wp[ 6]), p0706);
	muac63((vu[ 9], vu[ 8], wp[ 9], wp[ 8]), p0908);
	muac63((vu[11], vu[10], wp[11], wp[10]), p1110);
	muac63((vu[13], vu[12], wp[13], wp[12]), p1312);
	muac63((vu[15], vu[14], wp[15], wp[14]), p1514);

	muac63((vu[ 1], vu[ 0], wn[ 1], wn[ 0]), n0100);
	muac63((vu[ 3], vu[ 2], wn[ 3], wn[ 2]), n0302);
	muac63((vu[ 5], vu[ 4], wn[ 5], wn[ 4]), n0504);
	muac63((vu[ 7], vu[ 6], wn[ 7], wn[ 6]), n0706);
	muac63((vu[ 9], vu[ 8], wn[ 9], wn[ 8]), n0908);
	muac63((vu[11], vu[10], wn[11], wn[10]), n1110);
	muac63((vu[13], vu[12], wn[13], wn[12]), n1312);
	muac63((vu[15], vu[14], wn[15], wn[14]), n1514);

	uint4_t p0, p1, p2, p3;
	uint4_t n0, n1, n2, n3;

	uac64((p0100, p0302), p0);
	uac64((p0504, p0706), p1);
	uac64((p0908, p1110), p2);
	uac64((p1312, p1514), p3);

	uac64((n0100, n0302), n0);
	uac64((n0504, n0706), n1);
	uac64((n0908, n1110), n2);
	uac64((n1312, n1514), n3);

	return ((p0 + p1) + (p2 + p3)) - ((n0 + n1) + (n2 + n3));
}

void norm32(uint3_t i, uint2_t& o) {
	static const uint2_t table[] {
		0, 1, 2, 2, 3, 3, 3, 3,
	};
	o = table[i];
}

template <int F>
uint2_t batch_norm(int16_t acc, int thr[]) {
#pragma HLS inline
	uint2_t m;
	bit_t b0 = acc >= thr[0];
	bit_t b1 = acc >= thr[1];
	bit_t b2 = acc >= thr[2];
	norm32((b2, b1, b0), m);
	return m;
}

template <int ROWS, int COLS, typename T, typename WT>
class Window {
private:
	WT buf;
public:
	void shift_pixels_left() {
#pragma HLS inline
		for (int i = 0; i < ROWS * COLS - 1; i++) {
#pragma HLS unroll
			buf[i] = buf[i + 1];
		}
	}

	void insert_right_col(const T value[ROWS]) {
#pragma HLS inline
		for (int i = 0; i < ROWS; i++) {
#pragma HLS unroll
			int idx = (i + 1) * COLS - 1;
			buf[idx] = value[i];
		}
	}

	WT& get_buf() {
		return buf;
	}
};

template <int KH, int W, typename T, typename WT>
class LineBuffer {
private:
	hls::vector<T, (KH - 1) * W> buf;
	Window<KH, KH, T, WT> window;

	void shift_pixels_up() {
#pragma HLS inline
		for (int i = 0; i < (KH - 1) * W - 1; i++) {
#pragma HLS unroll
			buf[i] = buf[i + 1];
		}
	}

	void insert_bottom_row(T value) {
#pragma HLS inline
		buf[(KH - 1) * W - 1] = value;
	}

	void get_col(T value[KH - 1]) {
#pragma HLS inline
		for (int i = 0; i < KH - 1; i++) {
#pragma HLS unroll
			value[i] = buf[i * W];
		}
	}
public:
	void insert_linebuf(const T v) {
		shift_pixels_up();
		insert_bottom_row(v);
	}

	void slide_window(const T v) {
		T rows[KH];
#pragma HLS array_partition variable=rows

		get_col(rows);
		rows[KH - 1] = v;
		shift_pixels_up();
		insert_bottom_row(v);

		window.shift_pixels_left();
		window.insert_right_col(rows);
	}

	WT& get_window() {
		return window.get_buf();
	}
};

template <typename WT, int H, int W, int C, int KH, int KW, int F, int M>
class Conv2D_0 {
private:
	static const int OH = H - KH + 1;
	static const int OW = W - KW + 1;
public:
	template <typename OT>
	void compute(fifo<WT>& ins, fifo<OT>& outs) {
		static WT fp[F] = {
I25(0x18c671c), I25(0x0c63189), I25(0x1808cce), I25(0x007e042),
I25(0x1871871), I25(0x00c0000), I25(0x10c6201), I25(0x0118c43),
I25(0x0188400), I25(0x1c80000), I25(0x0006180), I25(0x0f00002),
I25(0x00043f9), I25(0x0777312), I25(0x1c7863a), I25(0x1f384a0),
		};
		static WT fn[F] = {
I25(0x0339001), I25(0x039ce34), I25(0x0277311), I25(0x1f01fa9),
I25(0x038e70c), I25(0x0403783), I25(0x0f39cce), I25(0x1ee739c),
I25(0x0e71bff), I25(0x0377ff3), I25(0x1f78c23), I25(0x10dffd8),
I25(0x1ffa800), I25(0x1888cec), I25(0x02871c0), I25(0x0006b1f),
		};
		static int thr[F][M] = {
{ 9, 11, 13, }, { 8, 9, 11, }, { 6, 8, 11, }, { 7, 8, 10, },
{ 6, 8, 10, }, { 24, 30, 36, }, { 10, 12, 14, }, { 10, 13, 16, },
{ 14, 15, 17, }, { 14, 16, 18, }, { 11, 12, 14, }, { 11, 13, 14, },
{ 5, 8, 11, }, { 5, 7, 10, }, { 5, 7, 9, }, { 4, 7, 10, },
		};
#pragma HLS array_partition variable=fp cyclic factor=F
#pragma HLS array_partition variable=fn cyclic factor=F
#pragma HLS array_partition variable=thr cyclic factor=M dim=1

		for (int xy = 0; xy < OH * OW; xy++) {
#pragma HLS pipeline
			WT val = ins.read();
			OT oval;
			for (int z = 0; z < F; z++) {
				int16_t acc = muladd25(val, fp[z], fn[z]);
				oval[z] = batch_norm<F>(acc, thr[z]);
			}
			outs.write(oval);
		}
	}
};

template <typename WT, int H, int W, int C, int KH, int KW, int F, int M>
class Conv2D_1 {
protected:
	static const int OH = H - KH + 1;
	static const int OW = W - KW + 1;
public:
	template <typename OT>
	void compute(fifo<WT>& ins, fifo<OT>& outs) {
		static int_t<1,16> fp[F * KH * KW] = {
I16(0x0000), I16(0x2050), I16(0x0253), I16(0xc9c3),
I16(0x8912), I16(0x4812), I16(0x6641), I16(0x2348),
I16(0x3350), I16(0xc638), I16(0x8a9c), I16(0x48b9),
I16(0x0503), I16(0x0001), I16(0xc508), I16(0xd4e5),
I16(0x5065), I16(0x1005), I16(0x0402), I16(0x0080),
I16(0x3438), I16(0x4851), I16(0x4443), I16(0x0842),
I16(0x818b), I16(0x2022), I16(0x0048), I16(0x0804),
I16(0x2412), I16(0x2482), I16(0x0dc9), I16(0x5784),
I16(0x6e58), I16(0xc40a), I16(0xaa00), I16(0x9f26),
I16(0xd408), I16(0x9408), I16(0x8081), I16(0x2009),
I16(0x1008), I16(0x1021), I16(0x120b), I16(0x22cf),
I16(0x2462), I16(0x3001), I16(0x21d1), I16(0x25e3),
I16(0x35de), I16(0x148b), I16(0x91a1), I16(0x0597),
I16(0x4380), I16(0x4a80), I16(0x0441), I16(0xc48c),
I16(0x152c), I16(0x8404), I16(0x61c0), I16(0x2284),
I16(0x22af), I16(0x9004), I16(0xeb04), I16(0xa201),
I16(0x2008), I16(0x2280), I16(0x5200), I16(0x2200),
I16(0xaa02), I16(0x2804), I16(0x886c), I16(0xa2a4),
I16(0x2841), I16(0x29ec), I16(0xed88), I16(0x3018),
I16(0x4050), I16(0x2310), I16(0xa2e6), I16(0x83b6),
I16(0x209d), I16(0x0803), I16(0x02f1), I16(0x20ec),
I16(0x610d), I16(0x2085), I16(0x2205), I16(0x0144),
I16(0x812c), I16(0xe95c), I16(0x6824), I16(0x301b),
I16(0xd410), I16(0x403a), I16(0x6500), I16(0x1608),
I16(0x1400), I16(0x1e00), I16(0x1009), I16(0x3c0d),
I16(0x0a01), I16(0x8108), I16(0x41c0), I16(0x2b88),
I16(0x04c5), I16(0xe20c), I16(0xe204), I16(0xa084),
I16(0x8aac), I16(0x2abb), I16(0x6d04), I16(0x3900),
I16(0xc020), I16(0xc82c), I16(0xc97a), I16(0x80e0),
I16(0x2000), I16(0xd050), I16(0x584b), I16(0xd048),
I16(0x088c), I16(0x0057), I16(0x5096), I16(0x1540),
I16(0x104b), I16(0x0286), I16(0x5216), I16(0xc405),
I16(0x1547), I16(0x0348), I16(0x4084), I16(0x0900),
I16(0x0200), I16(0x5430), I16(0x1440), I16(0x0104),
I16(0x0200), I16(0x800a), I16(0xc810), I16(0x4210),
I16(0xaa00), I16(0x0819), I16(0xc041), I16(0xd7c8),
I16(0xce2a), I16(0xaa54), I16(0xbbe9), I16(0xcc9a),
I16(0x9d9a), I16(0xb080), I16(0x0ea4), I16(0x85ee),
I16(0x82ac), I16(0x4c0c), I16(0x1804), I16(0x243c),
I16(0xb420), I16(0xcc60), I16(0xc118), I16(0x5810),
I16(0x3e41), I16(0x5008), I16(0x0408), I16(0x0512),
I16(0x8192), I16(0x054a), I16(0x2118), I16(0x220c),
I16(0xa144), I16(0x87c2), I16(0xa48e), I16(0x3400),
I16(0x2240), I16(0x22ec), I16(0x00cf), I16(0x2449),
I16(0x1470), I16(0xab48), I16(0x0d40), I16(0xd812),
I16(0x500c), I16(0xf458), I16(0x1d1b), I16(0x4999),
I16(0x8103), I16(0x26ca), I16(0x8408), I16(0x4509),
I16(0x6181), I16(0x2080), I16(0x8e1a), I16(0x601a),
I16(0x208c), I16(0x2084), I16(0x2284), I16(0xa41a),
I16(0x3008), I16(0x10bd), I16(0x501c), I16(0x1914),
I16(0x2180), I16(0x08af), I16(0x9491), I16(0x0033),
I16(0x883f), I16(0x1204), I16(0x02a4), I16(0xc209),
I16(0xc201), I16(0x8a20), I16(0x6a24), I16(0x2340),
I16(0x2504), I16(0x3c28), I16(0x8001), I16(0x01c1),
I16(0x6120), I16(0x6026), I16(0x000e), I16(0x2192),
I16(0x0c63), I16(0x44a2), I16(0x9cfb), I16(0x6728),
I16(0x3314), I16(0x48cb), I16(0x0046), I16(0x0162),
I16(0x1a2c), I16(0x486d), I16(0x89d2), I16(0x404a),
I16(0x6224), I16(0x4a34), I16(0xbe1c), I16(0xc046),
I16(0xa002), I16(0xe032), I16(0x500c), I16(0x1c0c),
I16(0xa1d0), I16(0x6004), I16(0x40ba), I16(0xd41a),
I16(0x158a), I16(0x60f6), I16(0x808a), I16(0x8792),
I16(0x419a), I16(0xc511), I16(0x0304), I16(0x2a04),
I16(0x2b00), I16(0x08c1), I16(0x4100), I16(0x25c6),
I16(0x0642), I16(0x29c0), I16(0x018a), I16(0x078e),
I16(0x22c3), I16(0x08c8), I16(0x2d8e), I16(0x2728),
I16(0x42e4), I16(0x0742), I16(0x0162), I16(0x0102),
I16(0xc4a0), I16(0x42a6), I16(0x6d00), I16(0x09ce),
I16(0x0006), I16(0x01c4), I16(0x64c4), I16(0x5702),
I16(0x02bb), I16(0xe012), I16(0xaa5b), I16(0x01d6),
I16(0xa00a), I16(0x03c4), I16(0x8aa0), I16(0x3c8a),
I16(0x284a), I16(0x8020), I16(0x21bc), I16(0x048a),
I16(0x401e), I16(0x0586), I16(0x2001), I16(0x04f2),
I16(0xc13e), I16(0xed02), I16(0x32ac), I16(0x1c11),
I16(0x5402), I16(0x9c1a), I16(0xeb0d), I16(0x240c),
I16(0x8188), I16(0x0a43), I16(0x2544), I16(0x2718),
I16(0x4a0e), I16(0x8401), I16(0x8250), I16(0x9203),
I16(0x1031), I16(0x9a20), I16(0x5000), I16(0x1040),
I16(0xa548), I16(0x2d00), I16(0x1084), I16(0xd830),
I16(0x1410), I16(0x1210), I16(0x1104), I16(0x0824),
I16(0x0808), I16(0x8200), I16(0x0908), I16(0x6268),
I16(0xc000), I16(0x2a90), I16(0x2181), I16(0x07f8),
I16(0x12c4), I16(0x80c5), I16(0x07e8), I16(0x34c3),
I16(0x34c6), I16(0x0484), I16(0x0344), I16(0x1600),
I16(0x1c04), I16(0x5900), I16(0x0864), I16(0x6426),
I16(0x0a04), I16(0xa200), I16(0x8a28), I16(0xc948),
I16(0xc148), I16(0x2025), I16(0x2040), I16(0x0200),
I16(0x8681), I16(0x2190), I16(0x1d10), I16(0x1d00),
I16(0x0940), I16(0x4343), I16(0x0604), I16(0x465a),
I16(0x2390), I16(0x0940), I16(0x23c2), I16(0xa4c2),
I16(0x0646), I16(0x880a), I16(0xc4d5), I16(0x0542),
I16(0x0101), I16(0x0f12), I16(0x85c6), I16(0x8e10),
I16(0xc002), I16(0xd00b), I16(0xa43b), I16(0x8c49),
I16(0xb230), I16(0xe308), I16(0xaa48), I16(0x0450),
I16(0xc477), I16(0x0c3e), I16(0x2a88), I16(0x2688),
I16(0xc150), I16(0x1023), I16(0xd229), I16(0xa120),
I16(0x81cf), I16(0x3883), I16(0x5300), I16(0x5004),
I16(0x1a04), I16(0x18c5), I16(0x40e7), I16(0x01bb),
I16(0x4aba), I16(0x4800), I16(0x4086), I16(0x81ce),
I16(0x02ce), I16(0x85c2), I16(0x0141), I16(0x0811),
		};
		static int_t<1,16> fn[F * KH * KW] = {
I16(0xf7ed), I16(0x5f00), I16(0xb428), I16(0x160c),
I16(0x5441), I16(0xb040), I16(0x8010), I16(0x58a1),
I16(0x4c8e), I16(0x10c5), I16(0x3442), I16(0x0706),
I16(0x8abc), I16(0xf3c4), I16(0x1265), I16(0x080a),
I16(0x249a), I16(0xe730), I16(0xf0cc), I16(0x7c47),
I16(0x4046), I16(0x022e), I16(0xaa8c), I16(0xe10d),
I16(0x7854), I16(0x4401), I16(0x4010), I16(0xb490),
I16(0x4840), I16(0x1b4d), I16(0x7004), I16(0x2021),
I16(0x1004), I16(0x1841), I16(0x5154), I16(0x2099),
I16(0x23c2), I16(0x2b73), I16(0x475a), I16(0xcfd6),
I16(0x6bd7), I16(0xa054), I16(0x8c74), I16(0xcc10),
I16(0xd918), I16(0xc5ba), I16(0x9400), I16(0x9a10),
I16(0xca00), I16(0xeb00), I16(0x6618), I16(0xb068),
I16(0x9c7d), I16(0xb11a), I16(0x8a98), I16(0x2a73),
I16(0xc0c1), I16(0x2238), I16(0x8c32), I16(0xdc32),
I16(0x4410), I16(0x09b3), I16(0x1462), I16(0x5d9e),
I16(0x0db5), I16(0xd411), I16(0x8016), I16(0x5856),
I16(0x5434), I16(0x1673), I16(0x3093), I16(0x5c00),
I16(0x569e), I16(0x1613), I16(0x0271), I16(0x03c4),
I16(0x1628), I16(0x1849), I16(0x0d11), I16(0x1449),
I16(0xc500), I16(0x94e8), I16(0x1c08), I16(0x0c12),
I16(0x02c2), I16(0xc270), I16(0xd918), I16(0x7e89),
I16(0x1a80), I16(0x1480), I16(0x0481), I16(0x80c4),
I16(0x2385), I16(0x3b45), I16(0x90c2), I16(0xc0e7),
I16(0xc8de), I16(0xe0df), I16(0xe3e6), I16(0xc272),
I16(0xd5d2), I16(0x56c3), I16(0x9e19), I16(0xd413),
I16(0xc10a), I16(0x0c51), I16(0x15d3), I16(0x0e5b),
I16(0x3543), I16(0x5500), I16(0x9282), I16(0x40c1),
I16(0x04c7), I16(0x32c1), I16(0x1400), I16(0x5e09),
I16(0xcab5), I16(0x2da6), I16(0x22a4), I16(0x2216),
I16(0x8251), I16(0x8908), I16(0xa640), I16(0xeaaf),
I16(0xa3a4), I16(0xad48), I16(0x2969), I16(0x234a),
I16(0xa8b0), I16(0x6490), I16(0x9b53), I16(0xa06b),
I16(0xa54f), I16(0x82cf), I16(0xa31f), I16(0x96e9),
I16(0x1497), I16(0x7085), I16(0x25cf), I16(0x2dc9),
I16(0x55b6), I16(0x46c2), I16(0x228e), I16(0x2016),
I16(0x2005), I16(0x5403), I16(0x4004), I16(0x2200),
I16(0x6005), I16(0x0a47), I16(0x9048), I16(0x1810),
I16(0x5113), I16(0x9343), I16(0xe173), I16(0x4ac3),
I16(0x005f), I16(0x3384), I16(0x3641), I16(0x246f),
I16(0xc192), I16(0x85c3), I16(0x9bd1), I16(0x7a24),
I16(0x324d), I16(0xa091), I16(0x9040), I16(0xdc83),
I16(0x4620), I16(0x7808), I16(0x4201), I16(0x48c7),
I16(0x40b1), I16(0xdd12), I16(0x3710), I16(0x0082),
I16(0x0a83), I16(0x5083), I16(0x0004), I16(0x23c9),
I16(0x8071), I16(0x0381), I16(0x0080), I16(0x2224),
I16(0x7eec), I16(0x0800), I16(0x22d2), I16(0x3200),
I16(0x1a52), I16(0x5c1d), I16(0x2020), I16(0x9ce0),
I16(0xdf53), I16(0x1d7b), I16(0xdd59), I16(0x4ac5),
I16(0x0b42), I16(0xa340), I16(0xa7c3), I16(0x24e3),
I16(0xc61b), I16(0xe310), I16(0x2b4c), I16(0x3600),
I16(0x34c0), I16(0xec5b), I16(0x415b), I16(0x1494),
I16(0x14f6), I16(0x5596), I16(0x9481), I16(0xd497),
I16(0x18d2), I16(0x83d7), I16(0x3ec4), I16(0xfc00),
I16(0x9413), I16(0x9780), I16(0x3c51), I16(0x5424),
I16(0xb28c), I16(0x3a04), I16(0x6304), I16(0x8807),
I16(0x4821), I16(0xb204), I16(0xb408), I16(0x4e80),
I16(0x0451), I16(0x0582), I16(0x3200), I16(0x9c81),
I16(0x0591), I16(0x0543), I16(0x0163), I16(0x13b8),
I16(0x5bed), I16(0x1f81), I16(0x8bc3), I16(0xa373),
I16(0x1e2e), I16(0x97c0), I16(0xb101), I16(0x2061),
I16(0xa270), I16(0x1e00), I16(0x7a45), I16(0x3045),
I16(0xaa64), I16(0x32e6), I16(0xfc3b), I16(0x94ba),
I16(0xd4b9), I16(0x9026), I16(0xba6e), I16(0xd011),
I16(0x91bc), I16(0x923c), I16(0x1251), I16(0x1851),
I16(0xd834), I16(0x3027), I16(0x5201), I16(0x1017),
I16(0x301a), I16(0xd8bd), I16(0x5c00), I16(0x3a11),
I16(0x2257), I16(0x9448), I16(0x10be), I16(0xe620),
I16(0xf610), I16(0x5a19), I16(0x9139), I16(0x88fc),
I16(0x1440), I16(0x15ed), I16(0x40a0), I16(0xf801),
I16(0x5c80), I16(0x5820), I16(0x1111), I16(0x8311),
I16(0x5081), I16(0x7a9f), I16(0xdc00), I16(0x7a51),
I16(0x8d80), I16(0x4821), I16(0xd8fc), I16(0xb308),
I16(0x16c1), I16(0x10d0), I16(0x8941), I16(0xa1a4),
I16(0x2b9c), I16(0x03c5), I16(0x04f2), I16(0x5b53),
I16(0x6020), I16(0x90a4), I16(0x0a88), I16(0x1004),
I16(0x05d1), I16(0x78c4), I16(0x3585), I16(0x2418),
I16(0xa186), I16(0x61d3), I16(0x20b2), I16(0x428c),
I16(0x4a27), I16(0x42ef), I16(0xe671), I16(0x2348),
I16(0xeb27), I16(0xe1cd), I16(0xe443), I16(0xb5db),
I16(0x24f6), I16(0x24ef), I16(0xf497), I16(0x9997),
I16(0x1197), I16(0xd169), I16(0xde1c), I16(0xf000),
I16(0xc818), I16(0x2112), I16(0x9814), I16(0xc208),
I16(0xca29), I16(0xa22a), I16(0xd813), I16(0xe819),
I16(0x6153), I16(0xa0d1), I16(0xd513), I16(0x9099),
I16(0xd491), I16(0x5dd3), I16(0x3597), I16(0x1286),
I16(0x2297), I16(0xd78a), I16(0x1d97), I16(0x54d8),
I16(0x1842), I16(0x1046), I16(0x2067), I16(0xe0ef),
I16(0xa2b7), I16(0x903c), I16(0x71ba), I16(0x3805),
I16(0x1449), I16(0xf61e), I16(0xd80d), I16(0x5814),
I16(0x7821), I16(0x1674), I16(0x3222), I16(0x08bc),
I16(0xc414), I16(0xe0a4), I16(0x7809), I16(0x30c8),
I16(0x1891), I16(0x0880), I16(0x4880), I16(0x62a6),
I16(0x4041), I16(0x0023), I16(0x05b3), I16(0x1185),
I16(0x3280), I16(0x3101), I16(0x1434), I16(0x4053),
I16(0x008d), I16(0x2f98), I16(0x21d6), I16(0x4a92),
I16(0x4a10), I16(0x8244), I16(0xa06e), I16(0xa5db),
I16(0x80f3), I16(0xc310), I16(0x3e18), I16(0xae40),
I16(0x3044), I16(0x947d), I16(0xb719), I16(0x5021),
I16(0x5800), I16(0x5835), I16(0xea12), I16(0x778c),
		};
		static int thr[F][M] = {
{ 18, 25, 33, }, { 28, 34, 40, }, { 24, 33, 42, }, { 23, 30, 36, },
{ 21, 31, 41, }, { 15, 22, 29, }, { 22, 29, 36, }, { 24, 34, 43, },
{ 15, 23, 31, }, { 17, 27, 36, }, { 21, 30, 38, }, { 28, 36, 44, },
{ 7, 15, 23, }, { 16, 23, 31, }, { 34, 41, 48, }, { 22, 29, 35, },
		};
#pragma HLS array_partition variable=fp cyclic factor=KH * KW
#pragma HLS array_partition variable=fn cyclic factor=KH * KW
#pragma HLS array_partition variable=thr cyclic factor=M dim=1

		for (int xy = 0; xy < OH * OW; xy++) {
#pragma HLS pipeline
			WT val = ins.read();
			OT oval;
			for (int z = 0; z < F; z ++) {
				int16_t acc = 0;
				for (int k = 0; k < KH * KW; k++) {
					int_t<2,16> vu = val[k];
					int_t<1,16> wp = fp[z * KH * KW + k];
					int_t<1,16> wn = fn[z * KH * KW + k];
					acc += muluadd32(vu, wp, wn);
				}
				oval[z] = batch_norm<F>(acc, thr[z]);
			}
			outs.write(oval);
		}
	}
};

template <typename T, int H, int W, int C>
class MaxPool2x2 {
private:
	void maxpool(const T v1, const T v2, T& ov) {
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			ov[z] = (v1[z] > v2[z]) ? v1[z] : v2[z];
		}
	}
public:
	void compute_h(fifo<T>& ins, fifo<T>& outs) {
		for (int xy = 0; xy < H * W / 2; xy++) {
#pragma HLS pipeline
			T val1 = ins.read();
			T val2 = ins.read();
			T oval;
			maxpool(val1, val2, oval);
			outs.write(oval);
		}
	}

	void compute_v(fifo<T>& ins, fifo<T>& outs) {
		T buf[W / 2];
#pragma HLS array_partition variable=buf

		for (int y = 0; y < H / 2; y++) {
#pragma HLS pipeline
			for (int x = 0; x < W / 2; x++) {
				T val = ins.read();
				buf[x] = val;
			}
			for (int x = 0; x < W / 2; x++) {
				T val1 = buf[x];
				T val2 = ins.read();
				T oval;
				maxpool(val1, val2, oval);
				outs.write(oval);
			}
		}
	}
};

template <typename IT, int FL, int CL, int K>
class Dense {
public:
	void compute(fifo<IT>& ins, fifo<int16_t>& outs) {
		static int_t<1,16> matp[CL * FL / K] = {
I16(0x0306), I16(0x4021), I16(0x2060), I16(0x51e2),
I16(0x44e4), I16(0x0909), I16(0x0600), I16(0xf0f1),
I16(0x4b29), I16(0x0b0e), I16(0x2020), I16(0x8485),
I16(0x3060), I16(0xc0e2), I16(0x5c80), I16(0x822c),
I16(0x0404), I16(0x9256), I16(0x09a0), I16(0x032c),
I16(0x4871), I16(0x4405), I16(0x4049), I16(0x00c8),
I16(0x4c05), I16(0xbb21), I16(0x222c), I16(0x8063),
I16(0x0a80), I16(0x19e0), I16(0x4151), I16(0x048c),
I16(0x4000), I16(0x08cc), I16(0x4705), I16(0xbb18),
I16(0x2124), I16(0x60a6), I16(0x62dc), I16(0x5040),
I16(0x0647), I16(0x8006), I16(0x216a), I16(0x6022),
I16(0x018d), I16(0x40a8), I16(0x0201), I16(0x9672),
I16(0x0852), I16(0x0d88), I16(0x0500), I16(0x9107),
I16(0x3325), I16(0x0108), I16(0x4d3b), I16(0xd200),
I16(0xa614), I16(0xf692), I16(0x0bc2), I16(0x08ca),
I16(0x1801), I16(0x4406), I16(0x6029), I16(0x01c8),
I16(0x880f), I16(0xd260), I16(0x6b34), I16(0x800b),
I16(0x9284), I16(0x9483), I16(0x4041), I16(0x7011),
I16(0x431f), I16(0x8902), I16(0x0619), I16(0xb020),
I16(0x31f0), I16(0x486a), I16(0x0388), I16(0x2881),
I16(0x0e09), I16(0x10a6), I16(0x2384), I16(0x7020),
I16(0x945a), I16(0xf048), I16(0x0c09), I16(0x3022),
I16(0x09c6), I16(0x9c0a), I16(0x0c38), I16(0x6611),
I16(0x3fbc), I16(0x7088), I16(0x801a), I16(0xf0c0),
I16(0x4ac9), I16(0x2006), I16(0x0700), I16(0x982a),
I16(0x4824), I16(0x6408), I16(0x291c), I16(0x89c0),
I16(0x801f), I16(0x04e0), I16(0x79e0), I16(0xa407),
I16(0x9240), I16(0x8807), I16(0x3680), I16(0x790e),
I16(0x1a3e), I16(0x00e2), I16(0xa213), I16(0x0480),
I16(0x18a0), I16(0xa004), I16(0x3860), I16(0x6101),
I16(0x0c98), I16(0xb631), I16(0xbb48), I16(0x70b7),
I16(0x9104), I16(0x74f6), I16(0x0000), I16(0x2306),
I16(0x0919), I16(0x9026), I16(0xd808), I16(0x0721),
I16(0x7818), I16(0x70e4), I16(0x0612), I16(0x70f0),
I16(0x1008), I16(0x0603), I16(0x9948), I16(0x8026),
I16(0x7a24), I16(0x7508), I16(0x023c), I16(0x41e0),
I16(0x0213), I16(0x41c4), I16(0x8040), I16(0x8707),
I16(0x08e9), I16(0x0503), I16(0x3020), I16(0x10ba),
I16(0x1a3c), I16(0x62c2), I16(0xc00b), I16(0x00c0),
I16(0x18c2), I16(0x8700), I16(0x40c3), I16(0x450d),
		};
		static int_t<1,16> matn[CL * FL / K] = {
I16(0x78d8), I16(0x27d6), I16(0x8407), I16(0x0e09),
I16(0x1312), I16(0xb4f6), I16(0x79fa), I16(0x0808),
I16(0xb482), I16(0x74c0), I16(0x5cc9), I16(0x3a78),
I16(0x8605), I16(0x0e05), I16(0x8351), I16(0x01d2),
I16(0xf9f3), I16(0x0d09), I16(0xd41c), I16(0xe403),
I16(0x108e), I16(0xb9f8), I16(0x8e86), I16(0x8e34),
I16(0x9108), I16(0x44c6), I16(0xccd3), I16(0x3f88),
I16(0xa40b), I16(0xc00f), I16(0xb0ae), I16(0x9063),
I16(0xbf3c), I16(0xb311), I16(0xb0fa), I16(0x44c7),
I16(0x4a9b), I16(0x9c18), I16(0x1020), I16(0x8d0f),
I16(0x5038), I16(0x5ff9), I16(0x8e01), I16(0x8bdd),
I16(0xb062), I16(0xa656), I16(0xb0fa), I16(0x0988),
I16(0xf020), I16(0xf022), I16(0x587b), I16(0x6af8),
I16(0xc80a), I16(0x6234), I16(0xa0c4), I16(0x2026),
I16(0x198b), I16(0x0848), I16(0xa435), I16(0xc030),
I16(0xe7b8), I16(0x39d1), I16(0x9b54), I16(0x6427),
I16(0x7550), I16(0x2d8d), I16(0x948b), I16(0x2724),
I16(0x6563), I16(0x4214), I16(0xbf3c), I16(0x8eea),
I16(0x9040), I16(0x4685), I16(0x30e0), I16(0x4fc5),
I16(0xc60b), I16(0x3785), I16(0x7015), I16(0x9376),
I16(0xf1e0), I16(0x0d48), I16(0xc422), I16(0x8fdd),
I16(0x41a1), I16(0x0d05), I16(0xf1e6), I16(0x4ad9),
I16(0x7431), I16(0x63c1), I16(0x8343), I16(0x984a),
I16(0xc002), I16(0x8c15), I16(0x25a1), I16(0x011f),
I16(0x9126), I16(0x9bf8), I16(0x2875), I16(0x2755),
I16(0x835b), I16(0x99f6), I16(0x8403), I16(0x022e),
I16(0x5ec0), I16(0xaa1f), I16(0x8016), I16(0x5860),
I16(0x2d37), I16(0x1278), I16(0xc93f), I16(0x86d1),
I16(0x45c1), I16(0xb71d), I16(0x10e0), I16(0x9b1d),
I16(0xa213), I16(0x0a63), I16(0x001f), I16(0x9a70),
I16(0x2166), I16(0x09c8), I16(0x44a6), I16(0x0600),
I16(0x4e08), I16(0x0300), I16(0xfdf2), I16(0xd8b9),
I16(0xe464), I16(0x2a91), I16(0x23c6), I16(0x8852),
I16(0x04e3), I16(0x8e09), I16(0x79cd), I16(0x8f06),
I16(0xa5a4), I16(0xd8d8), I16(0x64b6), I16(0x3949),
I16(0x854b), I16(0x0a06), I16(0xb4c2), I16(0xba1d),
I16(0x5cec), I16(0x8c19), I16(0x612f), I16(0x58f8),
I16(0xe114), I16(0xd0fc), I16(0xce5d), I16(0x2b45),
I16(0x24c2), I16(0x8939), I16(0x16d4), I16(0xb23c),
I16(0x4739), I16(0x70fe), I16(0x8338), I16(0xbac2),
		};
#pragma HLS array_partition variable=matp cyclic factor=CL
#pragma HLS array_partition variable=matn cyclic factor=CL

		static int16_t acc[CL];
#pragma HLS array_partition variable=acc

		for (int i = 0; i < CL; i++) {
#pragma HLS unroll
			acc[i] = 0;
		}

		for (int j = 0; j < FL / K; j++) {
#pragma HLS pipeline
			IT vu = ins.read();
			for (int i = 0; i < CL; i++) {
				int_t<1,16> wp = matp[j * CL + i];
				int_t<1,16> wn = matn[j * CL + i];
				acc[i] += muluadd32(vu, wp, wn);
			}
		}

		for (int i = 0; i < CL; i++) {
#pragma HLS unroll
			outs.write(acc[i]);
		}
	}
};

using Conv0 = Conv2D_0<int_t<1,25>, 28, 28, 1, 5, 5, 16, 3>;
using MaxPool0 = MaxPool2x2<int_t<2,16>, 24, 24, 16>;
using Conv1 = Conv2D_1<wpack_t, 12, 12, 16, 5, 5, 16, 3>;
using MaxPool1 = MaxPool2x2<int_t<2,16>, 8, 8, 16>;
using MatMul0 = Dense<int_t<2,16>, 256, 10, 16>;

template <int H, int W, int KH, int KW>
void read_input(const int in[H * W], fifo<int_t<1,25>>& ins) {
	LineBuffer<KH, W, bit_t, int_t<1,25>> linebuf;

	int ptr = 0;
	for (int y = 0; y < KH - 1; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			bit_t v = in[ptr++];
			linebuf.insert_linebuf(v);
		}
	}
	for (int y = KH - 1; y < H; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			bit_t v = in[ptr++];
			linebuf.slide_window(v);

			if (x >= KW - 1) {
				int_t<1,25> oval = linebuf.get_window();
				ins.write(oval);
			}
		}
	}
}

template <int H, int W, int C, int KH, int KW>
void pass_through(fifo<int_t<2,16>>& ins, fifo<wpack_t>& outs) {
	LineBuffer<KH, W, int_t<2,16>, wpack_t> linebuf;

	for (int y = 0; y < KH - 1; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			int_t<2,16> val = ins.read();
			linebuf.insert_linebuf(val);
		}
	}
	for (int y = KH - 1; y < H; y++) {
#pragma HLS pipeline
		for (int x = 0; x < W; x++) {
			int_t<2,16> val = ins.read();
			linebuf.slide_window(val);

			if (x >= KW - 1) {
				wpack_t oval = linebuf.get_window();
				outs.write(oval);
			}
		}
	}
}

template <int CL>
void write_result(int out[1], fifo<int16_t>& outs) {
	int16_t max = -2000;
	int m = 0;

	for (int i = 0; i < CL; i++) {
#pragma HLS pipeline
		int16_t acc = outs.read();
		if (acc > max) {
			max = acc;
			m = i;
		}
	}

	out[0] = m;
}

void kernel(int in[HEIGHT * WIDTH], int out[1]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=WIDTH

	fifo<int_t<1,25>> ins("input_fifo");
	fifo<int_t<2,16>> pips1("pipe_fifo1");
	fifo<int_t<2,16>> pips2("pipe_fifo2");
	fifo<int_t<2,16>> pips3("pipe_fifo3");
	fifo<wpack_t> pips4("pipe_fifo4");
	fifo<int_t<2,16>> pips5("pipe_fifo5");
	fifo<int_t<2,16>> pips6("pipe_fifo6");
	fifo<int_t<2,16>> pips7("pipe_fifo7");
	fifo<int16_t> outs("output_fifo");

	Conv0 conv0;
	MaxPool0 maxpool0;
	Conv1 conv1;
	MaxPool1 maxpool1;
	MatMul0 matmul0;

#pragma HLS dataflow
	read_input<28, 28, 5, 5>(in, ins);
	conv0.compute<int_t<2,16>>(ins, pips1);
	maxpool0.compute_h(pips1, pips2);
	maxpool0.compute_v(pips2, pips3);
	pass_through<12, 12, 16, 5, 5>(pips3, pips4);
	conv1.compute<int_t<2,16>>(pips4, pips5);
	maxpool1.compute_h(pips5, pips6);
	maxpool1.compute_v(pips6, pips7);
	matmul0.compute(pips7, outs);
	write_result<10>(out, outs);
}
