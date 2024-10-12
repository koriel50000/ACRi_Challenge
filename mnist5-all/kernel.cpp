#include "kernel.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#define I16(i) int_t<1,16>({ (i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1, (i >> 4) & 1, (i >> 5) & 1, (i >> 6) & 1, (i >> 7) & 1, (i >> 8) & 1, (i >> 9) & 1, (i >> 10) & 1, (i >> 11) & 1, (i >> 12) & 1, (i >> 13) & 1, (i >> 14) & 1, (i >> 15) & 1, })
#define I25(i) int_t<1,25>({ (i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1, (i >> 4) & 1, (i >> 5) & 1, (i >> 6) & 1, (i >> 7) & 1, (i >> 8) & 1, (i >> 9) & 1, (i >> 10) & 1, (i >> 11) & 1, (i >> 12) & 1, (i >> 13) & 1, (i >> 14) & 1, (i >> 15) & 1, (i >> 16) & 1, (i >> 17) & 1, (i >> 18) & 1, (i >> 19) & 1, (i >> 20) & 1, (i >> 21) & 1, (i >> 22) & 1, (i >> 23) & 1, (i >> 24) & 1, })

const int WIDTH = 28;
const int HEIGHT = 28;

const int FILTER = 16;
const int KERNEL = 5;
const int CHANNEL = 16;
const int THRESHOLD = 3;

const int FLATTEN = 256;
const int CLASS = 10;

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
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		0, -1, 0, -1, 1, 0, 1, 0,
		0, -1, 0, -1, 1, 0, 1, 0,
		0, 0, -1, -1, 0, 0, -1, -1,
		1, 1, 0, 0, 1, 1, 0, 0,
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
I25(0x01bc800), I25(0x0008463), I25(0x0002598), I25(0x0462300),
I25(0x0d3b800), I25(0x011a000), I25(0x0004189), I25(0x1a48000),
I25(0x1001465), I25(0x00a6508), I25(0x10a4010), I25(0x0006502),
I25(0x00000ac), I25(0x0081095), I25(0x0310421), I25(0x1a08000),
		};
		static WT fn[F] = {
I25(0x000019e), I25(0x1ce7310), I25(0x0808001), I25(0x0018422),
I25(0x0004050), I25(0x0000000), I25(0x1e30000), I25(0x0001c48),
I25(0x0128000), I25(0x1000000), I25(0x0208002), I25(0x0100000),
I25(0x0f68000), I25(0x1f18000), I25(0x00c6100), I25(0x000436b),
		};
		static int thr[] = { 1, 3, 4 };
#pragma HLS array_partition variable=fp
#pragma HLS array_partition variable=fn
#pragma HLS array_partition variable=thr

		for (int xy = 0; xy < OH * OW; xy++) {
#pragma HLS pipeline
			WT val = ins.read();
			OT oval;
			for (int z = 0; z < F; z++) {
				int16_t acc = muladd25(val, fp[z], fn[z]);
				uint2_t m = 0;
				for (int n = 0; n < M; n++) {
					if (acc >= thr[n]) {
						m = n + 1;
					}
				}
				oval[z] = m;
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
I16(0x8000), I16(0x0000), I16(0x4000), I16(0x4000),
I16(0xc000), I16(0x1000), I16(0x1000), I16(0x0000),
I16(0x4000), I16(0x4000), I16(0x1000), I16(0x0000),
I16(0x0028), I16(0x0202), I16(0x0020), I16(0x0080),
I16(0x0008), I16(0x001a), I16(0x0002), I16(0x0000),
I16(0x0408), I16(0x0206), I16(0x0802), I16(0x0000),
I16(0x0001), I16(0x1000), I16(0x1002), I16(0x0000),
I16(0x0000), I16(0x0000), I16(0x9000), I16(0x9000),
I16(0x0000), I16(0x8000), I16(0x0000), I16(0x8081),
I16(0x9000), I16(0x0480), I16(0x8080), I16(0x0020),
I16(0x8221), I16(0x0088), I16(0x001d), I16(0x0004),
I16(0x0100), I16(0x0041), I16(0x1006), I16(0x6802),
I16(0x2c00), I16(0x0010), I16(0x0020), I16(0x1000),
I16(0x1000), I16(0x0000), I16(0x0000), I16(0x4030),
I16(0x8000), I16(0x1080), I16(0x0080), I16(0x0000),
I16(0x0120), I16(0x8020), I16(0x8080), I16(0x8081),
I16(0x8001), I16(0x3940), I16(0x2140), I16(0x3001),
I16(0x9001), I16(0x9001), I16(0x0040), I16(0x0040),
I16(0x0040), I16(0x1040), I16(0x1040), I16(0x4102),
I16(0x0082), I16(0x0000), I16(0x0000), I16(0x8000),
I16(0xc082), I16(0x0000), I16(0x0000), I16(0x0000),
I16(0x4000), I16(0x4002), I16(0x0000), I16(0x0000),
I16(0x0008), I16(0x0000), I16(0x0000), I16(0x0080),
I16(0x0088), I16(0x022a), I16(0x0002), I16(0x0000),
I16(0x0009), I16(0x002e), I16(0x0102), I16(0x0102),
I16(0x0000), I16(0x0000), I16(0x080c), I16(0x0304),
I16(0x0010), I16(0x0000), I16(0x0080), I16(0x0408),
I16(0x0400), I16(0x0000), I16(0x0002), I16(0x0000),
I16(0x0402), I16(0x023e), I16(0x0200), I16(0x0002),
I16(0x0000), I16(0x060a), I16(0x4222), I16(0x1000),
I16(0x0000), I16(0x0002), I16(0x4402), I16(0x0000),
I16(0x0080), I16(0x1100), I16(0x0000), I16(0x4000),
I16(0x0002), I16(0x0000), I16(0x1000), I16(0x0000),
I16(0x0082), I16(0x402a), I16(0x0002), I16(0x1000),
I16(0x0002), I16(0x400a), I16(0x400a), I16(0x4000),
I16(0x0000), I16(0x000a), I16(0x4002), I16(0x0118),
I16(0x4000), I16(0x0002), I16(0x0002), I16(0x4006),
I16(0x0200), I16(0x4010), I16(0x0080), I16(0x0008),
I16(0x0083), I16(0x8208), I16(0xb340), I16(0x0000),
I16(0x080c), I16(0x0022), I16(0x0040), I16(0x3004),
I16(0x0000), I16(0x2000), I16(0xe040), I16(0x9040),
I16(0x1000), I16(0x4000), I16(0x0000), I16(0xc000),
I16(0x9080), I16(0x8000), I16(0x0400), I16(0x8088),
I16(0x8831), I16(0x8021), I16(0x0000), I16(0x4000),
I16(0x0000), I16(0x0002), I16(0x0000), I16(0x0000),
I16(0x0002), I16(0x0002), I16(0x0002), I16(0x0002),
I16(0x0002), I16(0x0002), I16(0x0002), I16(0x448a),
I16(0x4002), I16(0x0002), I16(0x0002), I16(0x000a),
I16(0x000a), I16(0x00aa), I16(0x4000), I16(0x0002),
I16(0x0408), I16(0x4003), I16(0x0e02), I16(0xc018),
I16(0x0008), I16(0x0002), I16(0x0002), I16(0x0000),
I16(0x0000), I16(0x0082), I16(0x4082), I16(0x0000),
I16(0x0000), I16(0x8000), I16(0x0000), I16(0x0016),
I16(0x4031), I16(0x8020), I16(0x8091), I16(0x0000),
I16(0x0000), I16(0x0020), I16(0x1081), I16(0x9245),
I16(0x0000), I16(0x0000), I16(0x2000), I16(0xb044),
I16(0xb040), I16(0x0000), I16(0xc001), I16(0x0820),
I16(0x0a40), I16(0x1b04), I16(0x0400), I16(0x0802),
I16(0x2002), I16(0x0100), I16(0x0100), I16(0x0008),
I16(0x0022), I16(0x0000), I16(0x0000), I16(0x0000),
I16(0x0200), I16(0x0100), I16(0x0042), I16(0x0000),
I16(0x0000), I16(0x400a), I16(0x0002), I16(0x4080),
I16(0x8080), I16(0x8489), I16(0x0000), I16(0x0040),
I16(0xb240), I16(0x98c5), I16(0x9344), I16(0x0040),
I16(0x9040), I16(0x9040), I16(0x1040), I16(0x1000),
I16(0x9040), I16(0x9041), I16(0x90c1), I16(0x9000),
I16(0x8000), I16(0x9000), I16(0x8000), I16(0x8001),
I16(0x0001), I16(0x0000), I16(0x80a0), I16(0x40a2),
I16(0x002a), I16(0x0000), I16(0x0000), I16(0x0002),
I16(0x0002), I16(0x0002), I16(0x0002), I16(0x0008),
I16(0x8000), I16(0x1002), I16(0x0002), I16(0x0000),
I16(0x0000), I16(0x0021), I16(0x9080), I16(0x0000),
I16(0x0081), I16(0x0828), I16(0x0100), I16(0x0080),
I16(0x0089), I16(0x040e), I16(0x0022), I16(0x0100),
I16(0x0000), I16(0x040a), I16(0x0c12), I16(0x0002),
I16(0x1804), I16(0x1040), I16(0x0000), I16(0x4000),
I16(0x4000), I16(0x1000), I16(0x1000), I16(0x0080),
I16(0x0222), I16(0x0002), I16(0x0082), I16(0x0082),
I16(0x4002), I16(0x080a), I16(0x4002), I16(0x000a),
I16(0x0006), I16(0x400a), I16(0x440a), I16(0x4000),
I16(0x0000), I16(0x4002), I16(0x4002), I16(0x0600),
I16(0x0601), I16(0x1100), I16(0x0000), I16(0x0000),
I16(0x4000), I16(0x0000), I16(0x0000), I16(0x0000),
I16(0x0000), I16(0x0000), I16(0x4000), I16(0x0002),
I16(0x0002), I16(0x0002), I16(0x0002), I16(0x4000),
I16(0x0020), I16(0x0008), I16(0x000a), I16(0x0000),
I16(0x0080), I16(0x0021), I16(0x0210), I16(0x2018),
I16(0x0410), I16(0x8831), I16(0x80b0), I16(0x80c0),
I16(0x0080), I16(0x0001), I16(0x1000), I16(0x8001),
I16(0xb341), I16(0x0051), I16(0x4001), I16(0x0000),
I16(0x3044), I16(0x1044), I16(0x3140), I16(0x1040),
I16(0x0001), I16(0x0040), I16(0x1000), I16(0x1040),
I16(0x1040), I16(0x3800), I16(0x8000), I16(0x8000),
I16(0x9000), I16(0x1000), I16(0x1040), I16(0x0000),
I16(0x8082), I16(0x0000), I16(0x0000), I16(0x0000),
I16(0x0091), I16(0x40a0), I16(0x8000), I16(0x0000),
I16(0x0000), I16(0x2000), I16(0x00b0), I16(0x4091),
I16(0x8000), I16(0x4000), I16(0x1800), I16(0x2040),
I16(0x0011), I16(0x0011), I16(0x8091), I16(0x0000),
I16(0x0000), I16(0x1900), I16(0x2040), I16(0x0825),
		};
		static int_t<1,16> fn[F * KH * KW] = {
I16(0x0000), I16(0x0002), I16(0x0002), I16(0x1003),
I16(0x1040), I16(0x0000), I16(0x4002), I16(0x1000),
I16(0x1000), I16(0x1002), I16(0xc010), I16(0xc000),
I16(0x1000), I16(0x9000), I16(0x8480), I16(0xf040),
I16(0xb040), I16(0x8040), I16(0x9080), I16(0x8001),
I16(0x9040), I16(0x8000), I16(0x8001), I16(0x0080),
I16(0x0002), I16(0x4001), I16(0x4030), I16(0x0000),
I16(0x0021), I16(0x0000), I16(0x4400), I16(0x4208),
I16(0x0822), I16(0x5000), I16(0x1000), I16(0x0204),
I16(0x2042), I16(0x0040), I16(0x1002), I16(0x0000),
I16(0x0002), I16(0x0000), I16(0x9000), I16(0x9000),
I16(0x0000), I16(0x0002), I16(0x8080), I16(0x8080),
I16(0x0080), I16(0x100a), I16(0x4100), I16(0x4002),
I16(0x008a), I16(0x8210), I16(0x1400), I16(0x1002),
I16(0x0802), I16(0x0202), I16(0x0002), I16(0x0900),
I16(0x0002), I16(0x0002), I16(0x0002), I16(0x0000),
I16(0x4000), I16(0xc082), I16(0x000a), I16(0x0002),
I16(0x0000), I16(0x4000), I16(0x4082), I16(0x04aa),
I16(0x0088), I16(0x4098), I16(0x401a), I16(0x0000),
I16(0x0000), I16(0x4100), I16(0x0042), I16(0x3042),
I16(0x0000), I16(0x0000), I16(0x0000), I16(0x0002),
I16(0x1000), I16(0x0000), I16(0x8018), I16(0x0000),
I16(0x9000), I16(0x1040), I16(0x0208), I16(0xa044),
I16(0x9040), I16(0x9000), I16(0x9080), I16(0x9046),
I16(0x9142), I16(0x8000), I16(0x8001), I16(0x8080),
I16(0x015a), I16(0x4000), I16(0x8000), I16(0x0000),
I16(0x0000), I16(0x1998), I16(0x4021), I16(0x8000),
I16(0x8000), I16(0x9000), I16(0x2054), I16(0x0160),
I16(0x0040), I16(0x8000), I16(0x8080), I16(0x201c),
I16(0x4140), I16(0x1040), I16(0x8081), I16(0x8081),
I16(0x0800), I16(0x0000), I16(0x0080), I16(0x0000),
I16(0x0000), I16(0x0000), I16(0x0002), I16(0x1040),
I16(0x1140), I16(0x1044), I16(0x000a), I16(0x0000),
I16(0x1040), I16(0x1040), I16(0x0000), I16(0x4000),
I16(0x9040), I16(0x9000), I16(0x9040), I16(0xb400),
I16(0xb000), I16(0x9000), I16(0x9001), I16(0x9080),
I16(0x3044), I16(0x8040), I16(0x8000), I16(0x9001),
I16(0x1002), I16(0x0000), I16(0xb340), I16(0x3140),
I16(0x3000), I16(0x0000), I16(0x0000), I16(0x9060),
I16(0x9000), I16(0x0000), I16(0x0000), I16(0x4410),
I16(0x9121), I16(0x1000), I16(0x0080), I16(0x0000),
I16(0x0e33), I16(0x10c1), I16(0x0080), I16(0x0000),
I16(0x0002), I16(0x4d46), I16(0x0100), I16(0x0000),
I16(0x0000), I16(0x0000), I16(0x4008), I16(0x0000),
I16(0x0001), I16(0x0020), I16(0x0010), I16(0x1080),
I16(0x0000), I16(0x0201), I16(0x9165), I16(0x1040),
I16(0x1040), I16(0x0000), I16(0x0000), I16(0x1040),
I16(0x1040), I16(0x0001), I16(0x3854), I16(0x1040),
I16(0x9040), I16(0x1000), I16(0x1040), I16(0x2800),
I16(0x8000), I16(0x1000), I16(0x1000), I16(0x1000),
I16(0x3154), I16(0x1140), I16(0x9000), I16(0x0000),
I16(0x0689), I16(0x3040), I16(0x0000), I16(0x0000),
I16(0x0000), I16(0x0e0d), I16(0x1000), I16(0x1040),
I16(0x9000), I16(0x1000), I16(0x4004), I16(0x8002),
I16(0x1002), I16(0x0000), I16(0x0000), I16(0x4000),
I16(0x8001), I16(0x4082), I16(0x008a), I16(0x0088),
I16(0x0488), I16(0x0022), I16(0x1040), I16(0x1100),
I16(0x4000), I16(0x4000), I16(0x0022), I16(0x0000),
I16(0x0080), I16(0x0080), I16(0x4491), I16(0x9003),
I16(0x0000), I16(0x1480), I16(0x0e98), I16(0x0b28),
I16(0x1000), I16(0x8080), I16(0x8080), I16(0x10b1),
I16(0x0960), I16(0x8000), I16(0x0000), I16(0x0008),
I16(0x2100), I16(0x1040), I16(0x5102), I16(0x0000),
I16(0x0000), I16(0x0000), I16(0x0002), I16(0x000a),
I16(0x0498), I16(0x443a), I16(0x400b), I16(0x4631),
I16(0x0802), I16(0x0c0a), I16(0x0902), I16(0x4806),
I16(0x0a44), I16(0x4006), I16(0x0004), I16(0x0044),
I16(0x0002), I16(0x0000), I16(0x0000), I16(0x0000),
I16(0x0000), I16(0x0080), I16(0x0281), I16(0xc021),
I16(0x8081), I16(0x8690), I16(0x9120), I16(0x9141),
I16(0x0900), I16(0x0a00), I16(0x0a14), I16(0x5240),
I16(0x1141), I16(0x4000), I16(0x0008), I16(0x6040),
I16(0x2042), I16(0x0000), I16(0x0000), I16(0x0002),
I16(0x0000), I16(0x0000), I16(0x8000), I16(0x4002),
I16(0x0000), I16(0x0000), I16(0x0000), I16(0x1080),
I16(0x4022), I16(0x40a2), I16(0x0000), I16(0x0000),
I16(0x3040), I16(0x4202), I16(0x4000), I16(0x0040),
I16(0x1040), I16(0x1000), I16(0x4950), I16(0x1040),
I16(0x9040), I16(0x9040), I16(0x9000), I16(0x4100),
I16(0x9000), I16(0x9040), I16(0x9081), I16(0x9040),
I16(0xc010), I16(0x8081), I16(0x9001), I16(0x1000),
I16(0x0002), I16(0x0002), I16(0x0008), I16(0x1080),
I16(0x9004), I16(0x1944), I16(0x4000), I16(0x0000),
I16(0x1040), I16(0x1040), I16(0x2000), I16(0x8000),
I16(0x9000), I16(0x9040), I16(0x9000), I16(0x9000),
I16(0x2040), I16(0x8000), I16(0x8000), I16(0x9040),
I16(0x3000), I16(0x0040), I16(0x8000), I16(0x8000),
I16(0x0002), I16(0x0002), I16(0x6002), I16(0x0002),
I16(0x0002), I16(0x0302), I16(0x2d02), I16(0x4002),
I16(0x0002), I16(0x0002), I16(0x0002), I16(0x0002),
I16(0x4002), I16(0x4002), I16(0x4002), I16(0x0002),
I16(0x0000), I16(0x0422), I16(0x000b), I16(0x0002),
I16(0x0020), I16(0x0000), I16(0x4002), I16(0x4402),
I16(0x4002), I16(0x0131), I16(0x8020), I16(0x1040),
I16(0x2100), I16(0x4000), I16(0x4000), I16(0x8084),
I16(0x0000), I16(0x3044), I16(0x0000), I16(0x0400),
I16(0x2683), I16(0x0002), I16(0x0002), I16(0x1040),
I16(0x3040), I16(0x2a4c), I16(0x8002), I16(0x0002),
I16(0x1002), I16(0x0002), I16(0x0044), I16(0x0033),
I16(0x0003), I16(0x0082), I16(0x4002), I16(0x0002),
		};
		static int thr[M] = { 3, 9, 14 };
#pragma HLS array_partition variable=fp cyclic factor=KH * KW
#pragma HLS array_partition variable=fn cyclic factor=KH * KW
#pragma HLS array_partition variable=thr

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
				uint2_t m = 0;
				for (int n = 0; n < M; n++) {
					if (acc >= thr[n]) {
						m = n + 1;
					}
				}
				oval[z] = m;
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
I16(0x0000), I16(0x0000), I16(0x0244), I16(0x0444),
I16(0x9020), I16(0x00a0), I16(0x0000), I16(0x0100),
I16(0x1020), I16(0x0191), I16(0x0000), I16(0x0088),
I16(0x4000), I16(0x4000), I16(0x0000), I16(0x0100),
I16(0x0000), I16(0x8404), I16(0x0040), I16(0x0111),
I16(0x0000), I16(0x23c0), I16(0x4000), I16(0x0009),
I16(0x1090), I16(0x0404), I16(0x06c0), I16(0x8c06),
I16(0x4000), I16(0x4000), I16(0xc000), I16(0x0010),
I16(0x0000), I16(0x0000), I16(0x10d9), I16(0x4400),
I16(0x0400), I16(0x1c06), I16(0x4008), I16(0x8804),
I16(0x1023), I16(0x0000), I16(0x0440), I16(0x0400),
I16(0x1021), I16(0x0100), I16(0x30a0), I16(0x0640),
I16(0x0000), I16(0x9020), I16(0x0200), I16(0x1880),
I16(0x0400), I16(0x041c), I16(0x9824), I16(0x0200),
I16(0x0920), I16(0x0600), I16(0x8000), I16(0x0100),
I16(0x0000), I16(0x23e0), I16(0x1080), I16(0x1001),
I16(0x0980), I16(0x0600), I16(0x0900), I16(0x4000),
I16(0x0004), I16(0x0000), I16(0x5019), I16(0x0000),
I16(0x0090), I16(0x0000), I16(0x0151), I16(0x0400),
I16(0x8d0e), I16(0x4040), I16(0x0000), I16(0x2000),
I16(0xb021), I16(0x0000), I16(0x3803), I16(0x0000),
I16(0x0004), I16(0x0040), I16(0x10a0), I16(0x0000),
I16(0x1008), I16(0x0000), I16(0x8240), I16(0x1030),
I16(0x0003), I16(0x0010), I16(0x0000), I16(0x0440),
I16(0xb000), I16(0x0a80), I16(0x3001), I16(0x0000),
I16(0x0880), I16(0x0150), I16(0x8004), I16(0x0440),
I16(0x0000), I16(0x0400), I16(0x0008), I16(0x1012),
I16(0x0240), I16(0x1000), I16(0x0011), I16(0x0008),
I16(0x0980), I16(0x0000), I16(0x4000), I16(0x0000),
I16(0x200d), I16(0x0000), I16(0x0400), I16(0x0200),
I16(0x2001), I16(0x0800), I16(0x0000), I16(0x8546),
I16(0x0200), I16(0x8544), I16(0x0000), I16(0x0000),
I16(0x3020), I16(0x0200), I16(0x0904), I16(0x1030),
I16(0x0000), I16(0x0402), I16(0x0408), I16(0x0002),
I16(0x8000), I16(0x3828), I16(0xa004), I16(0x0480),
I16(0x0006), I16(0x0140), I16(0x4000), I16(0x1402),
I16(0x0020), I16(0x2800), I16(0x0004), I16(0x0040),
I16(0x2004), I16(0x0090), I16(0x0010), I16(0x0000),
I16(0x8d04), I16(0x0001), I16(0x0040), I16(0x0008),
I16(0x0004), I16(0x0440), I16(0x2021), I16(0x1008),
		};
		static int_t<1,16> matn[CL * FL / K] = {
I16(0x4000), I16(0x09e0), I16(0xb0a2), I16(0xb8ab),
I16(0x4004), I16(0x0040), I16(0x4504), I16(0x0200),
I16(0x0600), I16(0x0240), I16(0x8240), I16(0x0440),
I16(0x280b), I16(0x12a0), I16(0x4004), I16(0x0000),
I16(0x4404), I16(0x1238), I16(0x0400), I16(0x0240),
I16(0x04c8), I16(0x4004), I16(0x0900), I16(0x0000),
I16(0x4444), I16(0x9028), I16(0xc100), I16(0x0010),
I16(0x0280), I16(0x0690), I16(0x0608), I16(0xcc00),
I16(0x0800), I16(0x0480), I16(0xc404), I16(0x30b9),
I16(0x9115), I16(0x4000), I16(0x0000), I16(0x1010),
I16(0x4440), I16(0x147b), I16(0x1020), I16(0x38a1),
I16(0x0640), I16(0x0640), I16(0x4444), I16(0xb820),
I16(0x0202), I16(0x0400), I16(0xa504), I16(0x0740),
I16(0x0020), I16(0x0240), I16(0x0410), I16(0x0000),
I16(0x4400), I16(0xa100), I16(0x0a00), I16(0x0081),
I16(0x8987), I16(0x4404), I16(0x8004), I16(0x0448),
I16(0x0600), I16(0x30b9), I16(0x1000), I16(0x8306),
I16(0x1280), I16(0x0280), I16(0x0b42), I16(0x1da3),
I16(0x4004), I16(0x0082), I16(0x4000), I16(0x3039),
I16(0x0250), I16(0x0400), I16(0x8080), I16(0x0c48),
I16(0x4404), I16(0x0420), I16(0x0000), I16(0x1029),
I16(0x2c80), I16(0x1028), I16(0x0600), I16(0x3082),
I16(0x8344), I16(0x3c03), I16(0x4012), I16(0xc400),
I16(0x0240), I16(0x1929), I16(0x30d0), I16(0x1000),
I16(0x0400), I16(0x0100), I16(0x0544), I16(0x3281),
I16(0x9220), I16(0x4c0c), I16(0x0000), I16(0x0000),
I16(0x1400), I16(0x0000), I16(0x1200), I16(0xc004),
I16(0x1d9b), I16(0x0500), I16(0xc240), I16(0x01a1),
I16(0x1000), I16(0x0000), I16(0x2488), I16(0x0000),
I16(0x0040), I16(0x040a), I16(0x103b), I16(0x8d0e),
I16(0x0c00), I16(0x8500), I16(0x0400), I16(0x0010),
I16(0xa187), I16(0x0000), I16(0xa6c2), I16(0x8144),
I16(0x0600), I16(0x3124), I16(0x0090), I16(0x0004),
I16(0x1801), I16(0x1220), I16(0x8144), I16(0x0000),
I16(0x3eaa), I16(0xc504), I16(0x0c00), I16(0x8104),
I16(0x1020), I16(0x0400), I16(0x3039), I16(0x0040),
I16(0x0906), I16(0x0040), I16(0x36d9), I16(0xc404),
I16(0x0800), I16(0x4104), I16(0xa008), I16(0x0021),
I16(0x1219), I16(0x0000), I16(0x0006), I16(0x0040),
I16(0x3508), I16(0xa826), I16(0x0000), I16(0x4404),
		};
#pragma HLS array_partition variable=matp cyclic factor=CL
#pragma HLS array_partition variable=matn cyclic factor=CL

		for (int j = 0; j < FL / K; j++) {
#pragma HLS pipeline
			IT vu = ins.read();
			for (int i = 0; i < CL; i++) {
#pragma HLS unroll
				int_t<1,16> wp = matp[j * CL + i];
				int_t<1,16> wn = matn[j * CL + i];
				int16_t acc = muluadd32(vu, wp, wn);
				outs.write(acc);
			}
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

template <int FL, int CL, int K>
void write_result(int out[CL], fifo<int16_t>& outs) {
	static int16_t acc[CL];
#pragma HLS array_partition variable=acc

	for (int i = 0; i < CL; i++) {
#pragma HLS unroll
		acc[i] = 0;
	}

	for (int j = 0; j < FL / K; j++) {
#pragma HLS pipeline
		for (int i = 0; i < CL; i++) {
#pragma HLS unroll
			acc[i] += outs.read();
		}
	}

	for (int i = 0; i < CL; i++) {
#pragma HLS unroll
		out[i] = acc[i];
	}
}

// @TODO 入力[-1(0), 1(1)]にしてBrevitasのCNVで学習
// thresholdはどう決める？
void kernel(
	int in[HEIGHT * WIDTH],
	int conv0_weight[FILTER * KERNEL * KERNEL],
	int threshold0[THRESHOLD],
	int conv1_weight[FILTER * KERNEL * KERNEL * CHANNEL],
	int threshold1[THRESHOLD],
	int matmul0_weight[FLATTEN * CLASS],
	int out[CLASS])
{
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
	write_result<256, 10, 16>(out, outs);
}
