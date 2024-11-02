#include "kernel.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>

const int FLATTEN = 256;
const int CLASS = 10;

const int CHUNK_SIZE = 16;

using uint2_t = ap_uint<2>;
using uint3_t = ap_uint<3>;
using uint4_t = ap_uint<4>;
using uint6_t = ap_uint<6>;

template <typename T>
using fifo = hls::stream<T>;

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

template <int W, int N>
class int_t {
private:
	ap_uint<W*N> buf_;
public:
	int_t(const char* s) : buf_(s) {}
	int_t(unsigned int i) : buf_(i) {}
	int_t(unsigned long l) : buf_(l) {}

	ap_range_ref<W*N, false> operator[](size_t index) const {
		assert(index < N);
		return buf_(W * (N - index) - 1, W * (N - 1 - index));
	}

	ap_range_ref<W*N, false> operator[](size_t index) {
		assert(index < N);
		return buf_(W * (N - index) - 1, W * (N - 1 - index));
	}
};

template <typename IT, int FL, int CL, int K>
class Dense {
public:
	void compute(fifo<IT>& ins, fifo<int16_t>& outs) {
		static int_t<1,16> matp[CL * FL / K] = {
0x0000, 0x0000, 0x0244, 0x0444,
0x9020, 0x00a0, 0x0000, 0x0100,
0x1020, 0x0191, 0x0000, 0x0088,
0x4000, 0x4000, 0x0000, 0x0100,
0x0000, 0x8404, 0x0040, 0x0111,
0x0000, 0x23c0, 0x4000, 0x0009,
0x1090, 0x0404, 0x06c0, 0x8c06,
0x4000, 0x4000, 0xc000, 0x0010,
0x0000, 0x0000, 0x10d9, 0x4400,
0x0400, 0x1c06, 0x4008, 0x8804,
0x1023, 0x0000, 0x0440, 0x0400,
0x1021, 0x0100, 0x30a0, 0x0640,
0x0000, 0x9020, 0x0200, 0x1880,
0x0400, 0x041c, 0x9824, 0x0200,
0x0920, 0x0600, 0x8000, 0x0100,
0x0000, 0x23e0, 0x1080, 0x1001,
0x0980, 0x0600, 0x0900, 0x4000,
0x0004, 0x0000, 0x5019, 0x0000,
0x0090, 0x0000, 0x0151, 0x0400,
0x8d0e, 0x4040, 0x0000, 0x2000,
0xb021, 0x0000, 0x3803, 0x0000,
0x0004, 0x0040, 0x10a0, 0x0000,
0x1008, 0x0000, 0x8240, 0x1030,
0x0003, 0x0010, 0x0000, 0x0440,
0xb000, 0x0a80, 0x3001, 0x0000,
0x0880, 0x0150, 0x8004, 0x0440,
0x0000, 0x0400, 0x0008, 0x1012,
0x0240, 0x1000, 0x0011, 0x0008,
0x0980, 0x0000, 0x4000, 0x0000,
0x200d, 0x0000, 0x0400, 0x0200,
0x2001, 0x0800, 0x0000, 0x8546,
0x0200, 0x8544, 0x0000, 0x0000,
0x3020, 0x0200, 0x0904, 0x1030,
0x0000, 0x0402, 0x0408, 0x0002,
0x8000, 0x3828, 0xa004, 0x0480,
0x0006, 0x0140, 0x4000, 0x1402,
0x0020, 0x2800, 0x0004, 0x0040,
0x2004, 0x0090, 0x0010, 0x0000,
0x8d04, 0x0001, 0x0040, 0x0008,
0x0004, 0x0440, 0x2021, 0x1008,
		};
		static int_t<1,16> matn[CL * FL / K] = {
0x4000, 0x09e0, 0xb0a2, 0xb8ab,
0x4004, 0x0040, 0x4504, 0x0200,
0x0600, 0x0240, 0x8240, 0x0440,
0x280b, 0x12a0, 0x4004, 0x0000,
0x4404, 0x1238, 0x0400, 0x0240,
0x04c8, 0x4004, 0x0900, 0x0000,
0x4444, 0x9028, 0xc100, 0x0010,
0x0280, 0x0690, 0x0608, 0xcc00,
0x0800, 0x0480, 0xc404, 0x30b9,
0x9115, 0x4000, 0x0000, 0x1010,
0x4440, 0x147b, 0x1020, 0x38a1,
0x0640, 0x0640, 0x4444, 0xb820,
0x0202, 0x0400, 0xa504, 0x0740,
0x0020, 0x0240, 0x0410, 0x0000,
0x4400, 0xa100, 0x0a00, 0x0081,
0x8987, 0x4404, 0x8004, 0x0448,
0x0600, 0x30b9, 0x1000, 0x8306,
0x1280, 0x0280, 0x0b42, 0x1da3,
0x4004, 0x0082, 0x4000, 0x3039,
0x0250, 0x0400, 0x8080, 0x0c48,
0x4404, 0x0420, 0x0000, 0x1029,
0x2c80, 0x1028, 0x0600, 0x3082,
0x8344, 0x3c03, 0x4012, 0xc400,
0x0240, 0x1929, 0x30d0, 0x1000,
0x0400, 0x0100, 0x0544, 0x3281,
0x9220, 0x4c0c, 0x0000, 0x0000,
0x1400, 0x0000, 0x1200, 0xc004,
0x1d9b, 0x0500, 0xc240, 0x01a1,
0x1000, 0x0000, 0x2488, 0x0000,
0x0040, 0x040a, 0x103b, 0x8d0e,
0x0c00, 0x8500, 0x0400, 0x0010,
0xa187, 0x0000, 0xa6c2, 0x8144,
0x0600, 0x3124, 0x0090, 0x0004,
0x1801, 0x1220, 0x8144, 0x0000,
0x3eaa, 0xc504, 0x0c00, 0x8104,
0x1020, 0x0400, 0x3039, 0x0040,
0x0906, 0x0040, 0x36d9, 0xc404,
0x0800, 0x4104, 0xa008, 0x0021,
0x1219, 0x0000, 0x0006, 0x0040,
0x3508, 0xa826, 0x0000, 0x4404,
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

using MatMul0 = Dense<int_t<2,16>, 256, 10, 16>;

template <int FL, int K>
void read_input(const int in[FL], fifo<int_t<2,16>>& ins) {
	for (int j = 0; j < FL / K; j++) {
#pragma HLS pipeline
		int_t<2,16> val;
		for (int k = 0; k < K; k++) {
#pragma HLS unroll
			val[k] = in[j * K + k];
		}
		ins.write(val);
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

void kernel(int in[FLATTEN], int weight[CLASS * FLATTEN], int out[CLASS]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=out

	fifo<int_t<2,16>> ins("input_fifo");
	fifo<int16_t> outs("output_fifo");

	MatMul0 matmul0;

#pragma HLS dataflow
	read_input<256, 16>(in, ins);
	matmul0.compute(ins, outs);
	write_result<256, 10, 16>(out, outs);
}
