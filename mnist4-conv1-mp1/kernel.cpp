#include "kernel.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>

const int WIDTH = 12;
const int HEIGHT = 12;
const int CHANNEL = 16;

const int FILTER = 16;
const int KERNEL = 5;
const int THRESHOLD = 3;

const int OWIDTH = 4;
const int OHEIGHT = 4;

template <int W, int N>
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
		return buf_(W * index + W - 1, W * index); // FIXME buf_(W * (N - index) - 1, W * (N - index - 1));
	}

	inline ap_range_ref<W*N, false> operator[](size_t index) {
		assert(index < N);
		return buf_(W * index + W - 1, W * index); // FIXME buf_(W * (N - index) - 1, W * (N - index - 1));
	}
};

template <typename T>
using fifo = hls::stream<T>;

using uint2_t = ap_uint<2>;
using uint3_t = ap_uint<3>;
using uint4_t = ap_uint<4>;
using uint6_t = ap_uint<6>;
using wpack_t = hls::vector<int_t<2,16>, KERNEL * KERNEL>;

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
#pragma HLS pipeline
		int x = 0 - (KN - 1);
		int y = 0 - (KN - 1);
		for (int i = 0; i < (W + PD) * (H + PD * 2) + PD; i++) {
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

template <typename WT, int H, int W, int C, int KN, int F, int M>
class Conv2D_1 {
protected:
	static const int OH = H - KN + 1;
	static const int OW = W - KN + 1;
public:
	template <typename OT>
	void compute(fifo<WT>& ins, fifo<OT>& outs) {
		static int_t<1,16> fp[F * KN * KN] = {
0x8000, 0x0000, 0x4000, 0x4000, 
0xc000, 0x1000, 0x1000, 0x0000, 
0x4000, 0x4000, 0x1000, 0x0000, 
0x0028, 0x0202, 0x0020, 0x0080, 
0x0008, 0x001a, 0x0002, 0x0000, 
0x0408, 0x0206, 0x0802, 0x0000, 
0x0001, 0x1000, 0x1002, 0x0000, 
0x0000, 0x0000, 0x9000, 0x9000, 
0x0000, 0x8000, 0x0000, 0x8081, 
0x9000, 0x0480, 0x8080, 0x0020, 
0x8221, 0x0088, 0x001d, 0x0004, 
0x0100, 0x0041, 0x1006, 0x6802, 
0x2c00, 0x0010, 0x0020, 0x1000, 
0x1000, 0x0000, 0x0000, 0x4030, 
0x8000, 0x1080, 0x0080, 0x0000, 
0x0120, 0x8020, 0x8080, 0x8081, 
0x8001, 0x3940, 0x2140, 0x3001, 
0x9001, 0x9001, 0x0040, 0x0040, 
0x0040, 0x1040, 0x1040, 0x4102, 
0x0082, 0x0000, 0x0000, 0x8000, 
0xc082, 0x0000, 0x0000, 0x0000, 
0x4000, 0x4002, 0x0000, 0x0000, 
0x0008, 0x0000, 0x0000, 0x0080, 
0x0088, 0x022a, 0x0002, 0x0000, 
0x0009, 0x002e, 0x0102, 0x0102, 
0x0000, 0x0000, 0x080c, 0x0304, 
0x0010, 0x0000, 0x0080, 0x0408, 
0x0400, 0x0000, 0x0002, 0x0000, 
0x0402, 0x023e, 0x0200, 0x0002, 
0x0000, 0x060a, 0x4222, 0x1000, 
0x0000, 0x0002, 0x4402, 0x0000, 
0x0080, 0x1100, 0x0000, 0x4000, 
0x0002, 0x0000, 0x1000, 0x0000, 
0x0082, 0x402a, 0x0002, 0x1000, 
0x0002, 0x400a, 0x400a, 0x4000, 
0x0000, 0x000a, 0x4002, 0x0118, 
0x4000, 0x0002, 0x0002, 0x4006, 
0x0200, 0x4010, 0x0080, 0x0008, 
0x0083, 0x8208, 0xb340, 0x0000, 
0x080c, 0x0022, 0x0040, 0x3004, 
0x0000, 0x2000, 0xe040, 0x9040, 
0x1000, 0x4000, 0x0000, 0xc000, 
0x9080, 0x8000, 0x0400, 0x8088, 
0x8831, 0x8021, 0x0000, 0x4000, 
0x0000, 0x0002, 0x0000, 0x0000, 
0x0002, 0x0002, 0x0002, 0x0002, 
0x0002, 0x0002, 0x0002, 0x448a, 
0x4002, 0x0002, 0x0002, 0x000a, 
0x000a, 0x00aa, 0x4000, 0x0002, 
0x0408, 0x4003, 0x0e02, 0xc018, 
0x0008, 0x0002, 0x0002, 0x0000, 
0x0000, 0x0082, 0x4082, 0x0000, 
0x0000, 0x8000, 0x0000, 0x0016, 
0x4031, 0x8020, 0x8091, 0x0000, 
0x0000, 0x0020, 0x1081, 0x9245, 
0x0000, 0x0000, 0x2000, 0xb044, 
0xb040, 0x0000, 0xc001, 0x0820, 
0x0a40, 0x1b04, 0x0400, 0x0802, 
0x2002, 0x0100, 0x0100, 0x0008, 
0x0022, 0x0000, 0x0000, 0x0000, 
0x0200, 0x0100, 0x0042, 0x0000, 
0x0000, 0x400a, 0x0002, 0x4080, 
0x8080, 0x8489, 0x0000, 0x0040, 
0xb240, 0x98c5, 0x9344, 0x0040, 
0x9040, 0x9040, 0x1040, 0x1000, 
0x9040, 0x9041, 0x90c1, 0x9000, 
0x8000, 0x9000, 0x8000, 0x8001, 
0x0001, 0x0000, 0x80a0, 0x40a2, 
0x002a, 0x0000, 0x0000, 0x0002, 
0x0002, 0x0002, 0x0002, 0x0008, 
0x8000, 0x1002, 0x0002, 0x0000, 
0x0000, 0x0021, 0x9080, 0x0000, 
0x0081, 0x0828, 0x0100, 0x0080, 
0x0089, 0x040e, 0x0022, 0x0100, 
0x0000, 0x040a, 0x0c12, 0x0002, 
0x1804, 0x1040, 0x0000, 0x4000, 
0x4000, 0x1000, 0x1000, 0x0080, 
0x0222, 0x0002, 0x0082, 0x0082, 
0x4002, 0x080a, 0x4002, 0x000a, 
0x0006, 0x400a, 0x440a, 0x4000, 
0x0000, 0x4002, 0x4002, 0x0600, 
0x0601, 0x1100, 0x0000, 0x0000, 
0x4000, 0x0000, 0x0000, 0x0000, 
0x0000, 0x0000, 0x4000, 0x0002, 
0x0002, 0x0002, 0x0002, 0x4000, 
0x0020, 0x0008, 0x000a, 0x0000, 
0x0080, 0x0021, 0x0210, 0x2018, 
0x0410, 0x8831, 0x80b0, 0x80c0, 
0x0080, 0x0001, 0x1000, 0x8001, 
0xb341, 0x0051, 0x4001, 0x0000, 
0x3044, 0x1044, 0x3140, 0x1040, 
0x0001, 0x0040, 0x1000, 0x1040, 
0x1040, 0x3800, 0x8000, 0x8000, 
0x9000, 0x1000, 0x1040, 0x0000, 
0x8082, 0x0000, 0x0000, 0x0000, 
0x0091, 0x40a0, 0x8000, 0x0000, 
0x0000, 0x2000, 0x00b0, 0x4091, 
0x8000, 0x4000, 0x1800, 0x2040, 
0x0011, 0x0011, 0x8091, 0x0000, 
0x0000, 0x1900, 0x2040, 0x0825, 
		};
		static int_t<1,16> fn[F * KN * KN] = {
0x0000, 0x0002, 0x0002, 0x1003, 
0x1040, 0x0000, 0x4002, 0x1000, 
0x1000, 0x1002, 0xc010, 0xc000, 
0x1000, 0x9000, 0x8480, 0xf040, 
0xb040, 0x8040, 0x9080, 0x8001, 
0x9040, 0x8000, 0x8001, 0x0080, 
0x0002, 0x4001, 0x4030, 0x0000, 
0x0021, 0x0000, 0x4400, 0x4208, 
0x0822, 0x5000, 0x1000, 0x0204, 
0x2042, 0x0040, 0x1002, 0x0000, 
0x0002, 0x0000, 0x9000, 0x9000, 
0x0000, 0x0002, 0x8080, 0x8080, 
0x0080, 0x100a, 0x4100, 0x4002, 
0x008a, 0x8210, 0x1400, 0x1002, 
0x0802, 0x0202, 0x0002, 0x0900, 
0x0002, 0x0002, 0x0002, 0x0000, 
0x4000, 0xc082, 0x000a, 0x0002, 
0x0000, 0x4000, 0x4082, 0x04aa, 
0x0088, 0x4098, 0x401a, 0x0000, 
0x0000, 0x4100, 0x0042, 0x3042, 
0x0000, 0x0000, 0x0000, 0x0002, 
0x1000, 0x0000, 0x8018, 0x0000, 
0x9000, 0x1040, 0x0208, 0xa044, 
0x9040, 0x9000, 0x9080, 0x9046, 
0x9142, 0x8000, 0x8001, 0x8080, 
0x015a, 0x4000, 0x8000, 0x0000, 
0x0000, 0x1998, 0x4021, 0x8000, 
0x8000, 0x9000, 0x2054, 0x0160, 
0x0040, 0x8000, 0x8080, 0x201c, 
0x4140, 0x1040, 0x8081, 0x8081, 
0x0800, 0x0000, 0x0080, 0x0000, 
0x0000, 0x0000, 0x0002, 0x1040, 
0x1140, 0x1044, 0x000a, 0x0000, 
0x1040, 0x1040, 0x0000, 0x4000, 
0x9040, 0x9000, 0x9040, 0xb400, 
0xb000, 0x9000, 0x9001, 0x9080, 
0x3044, 0x8040, 0x8000, 0x9001, 
0x1002, 0x0000, 0xb340, 0x3140, 
0x3000, 0x0000, 0x0000, 0x9060, 
0x9000, 0x0000, 0x0000, 0x4410, 
0x9121, 0x1000, 0x0080, 0x0000, 
0x0e33, 0x10c1, 0x0080, 0x0000, 
0x0002, 0x4d46, 0x0100, 0x0000, 
0x0000, 0x0000, 0x4008, 0x0000, 
0x0001, 0x0020, 0x0010, 0x1080, 
0x0000, 0x0201, 0x9165, 0x1040, 
0x1040, 0x0000, 0x0000, 0x1040, 
0x1040, 0x0001, 0x3854, 0x1040, 
0x9040, 0x1000, 0x1040, 0x2800, 
0x8000, 0x1000, 0x1000, 0x1000, 
0x3154, 0x1140, 0x9000, 0x0000, 
0x0689, 0x3040, 0x0000, 0x0000, 
0x0000, 0x0e0d, 0x1000, 0x1040, 
0x9000, 0x1000, 0x4004, 0x8002, 
0x1002, 0x0000, 0x0000, 0x4000, 
0x8001, 0x4082, 0x008a, 0x0088, 
0x0488, 0x0022, 0x1040, 0x1100, 
0x4000, 0x4000, 0x0022, 0x0000, 
0x0080, 0x0080, 0x4491, 0x9003, 
0x0000, 0x1480, 0x0e98, 0x0b28, 
0x1000, 0x8080, 0x8080, 0x10b1, 
0x0960, 0x8000, 0x0000, 0x0008, 
0x2100, 0x1040, 0x5102, 0x0000, 
0x0000, 0x0000, 0x0002, 0x000a, 
0x0498, 0x443a, 0x400b, 0x4631, 
0x0802, 0x0c0a, 0x0902, 0x4806, 
0x0a44, 0x4006, 0x0004, 0x0044, 
0x0002, 0x0000, 0x0000, 0x0000, 
0x0000, 0x0080, 0x0281, 0xc021, 
0x8081, 0x8690, 0x9120, 0x9141, 
0x0900, 0x0a00, 0x0a14, 0x5240, 
0x1141, 0x4000, 0x0008, 0x6040, 
0x2042, 0x0000, 0x0000, 0x0002, 
0x0000, 0x0000, 0x8000, 0x4002, 
0x0000, 0x0000, 0x0000, 0x1080, 
0x4022, 0x40a2, 0x0000, 0x0000, 
0x3040, 0x4202, 0x4000, 0x0040, 
0x1040, 0x1000, 0x4950, 0x1040, 
0x9040, 0x9040, 0x9000, 0x4100, 
0x9000, 0x9040, 0x9081, 0x9040, 
0xc010, 0x8081, 0x9001, 0x1000, 
0x0002, 0x0002, 0x0008, 0x1080, 
0x9004, 0x1944, 0x4000, 0x0000, 
0x1040, 0x1040, 0x2000, 0x8000, 
0x9000, 0x9040, 0x9000, 0x9000, 
0x2040, 0x8000, 0x8000, 0x9040, 
0x3000, 0x0040, 0x8000, 0x8000, 
0x0002, 0x0002, 0x6002, 0x0002, 
0x0002, 0x0302, 0x2d02, 0x4002, 
0x0002, 0x0002, 0x0002, 0x0002, 
0x4002, 0x4002, 0x4002, 0x0002, 
0x0000, 0x0422, 0x000b, 0x0002, 
0x0020, 0x0000, 0x4002, 0x4402, 
0x4002, 0x0131, 0x8020, 0x1040, 
0x2100, 0x4000, 0x4000, 0x8084, 
0x0000, 0x3044, 0x0000, 0x0400, 
0x2683, 0x0002, 0x0002, 0x1040, 
0x3040, 0x2a4c, 0x8002, 0x0002, 
0x1002, 0x0002, 0x0044, 0x0033, 
0x0003, 0x0082, 0x4002, 0x0002, 
		};
		static int thr[M] = { 3, 9, 14 };
#pragma HLS array_partition variable=fp cyclic factor=KN * KN
#pragma HLS array_partition variable=fn cyclic factor=KN * KN
#pragma HLS array_partition variable=thr

		for (int xy = 0; xy < OH * OW; xy++) {
#pragma HLS pipeline
			WT val = ins.read();
			OT oval;
			for (int z = 0; z < F; z ++) {
				int16_t acc = 0;
				for (int k = 0; k < KN * KN; k++) {
					int_t<2,16> vu = val[k];
					int_t<1,16> wp = fp[z * KN * KN + k];
					int_t<1,16> wn = fn[z * KN * KN + k];
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

using Buffer1 = WindowBuffer<12, 12, 5, int_t<2,16>, wpack_t>;
using Conv1 = Conv2D_1<wpack_t, 12, 12, 16, 5, 16, 3>;
using MaxPool1 = MaxPool2x2<int_t<2,16>, 8, 8, 16>;

template<int H, int W, int C, typename T>
void read_input(const int in[H * W * C], fifo<T>& ins) {
	int ptr = 0;
	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS pipeline
		T val;
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			val[z] = in[ptr++];
		}
		ins.write(val);
	}
}

template<int H, int W, int C>
void write_result(int out[H * W * C], fifo<int_t<2,16>>& outs) {
	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS pipeline
		int_t<2,16> val = outs.read();
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			out[xy * C + z] = val[z];
		}
	}
}

void kernel(
	int in[HEIGHT * WIDTH * CHANNEL],
	int weight[FILTER * KERNEL * KERNEL * CHANNEL],
	int threshold[THRESHOLD],
	int out[OHEIGHT * OWIDTH * FILTER])
{
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHANNEL
#pragma HLS array_partition variable=out cyclic factor=FILTER

	fifo<int_t<2,16>> ins("input_fifo");
	fifo<wpack_t> pips1("pipe_fifo1");
	fifo<int_t<2,16>> pips2("pipe_fifo2");
	fifo<int_t<2,16>> pips3("pipe_fifo3");
	fifo<int_t<2,16>> outs("output_fifo");

	Buffer1 buffer1;
	Conv1 conv1;
	MaxPool1 maxpool1;

#pragma HLS dataflow
	read_input<12, 12, 16, int_t<2,16>>(in, ins);
	buffer1.pass_through(ins, pips1);
	conv1.compute<int_t<2,16>>(pips1, pips2);
	maxpool1.compute_h(pips2, pips3);
	maxpool1.compute_v(pips3, outs);
	write_result<4, 4, 16>(out, outs);
}
