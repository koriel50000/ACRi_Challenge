#include "kernel.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

const int FLATTEN = 256;
const int CLASS = 10;

const int CHUNK_SIZE = 16;

using int2_t = ap_int<2>;
using uint2_t = ap_uint<2>;
using int3_t = ap_int<3>;
using uint3_t = ap_uint<3>;
using int4_t = ap_int<4>;
using uint4_t = ap_uint<4>;
using uint6_t = ap_uint<6>;
using int1x16_t = ap_uint<1 * CHUNK_SIZE>;
using int2x16_t = ap_uint<2 * CHUNK_SIZE>;
template <typename T>
using fifo = hls::stream<T>;

namespace bit {
	template <int S>
	int2_t get(const ap_uint<2 * S>& src, const int idx) {
#pragma HLS inline
		int p = 2 * idx;
		return src(p + 2 - 1, p);
	}

	template <int S>
	void set(ap_uint<2 * S>& src, const int idx, const int2_t& v) {
#pragma HLS inline
		int p = 2 * idx;
		src(p + 2 - 1, p) = v;
	}

	template <int S>
	uint2_t getu(const ap_uint<2 * S>& src, const int idx) {
#pragma HLS inline
		int p = 2 * idx;
		return src(p + 2 - 1, p);
	}

	template <int S>
	void setu(ap_uint<2 * S>& src, const int idx, const uint2_t& v) {
#pragma HLS inline
		int p = 2 * idx;
		src(p + 2 - 1, p) = v;
	}
} // namespace bit

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

int16_t muluadd32(int2x16_t vu, int1x16_t wp, int1x16_t wn) {
	uint3_t p0300, p0704, p1108, p1512, p1916, p2320, p2724, p3128;
	uint3_t n0300, n0704, n1108, n1512, n1916, n2320, n2724, n3128;

	muac63((vu( 3,	0), wp( 1,  0)), p0300);
	muac63((vu( 7,	4), wp( 3,  2)), p0704);
	muac63((vu(11,	8), wp( 5,  4)), p1108);
	muac63((vu(15, 12), wp( 7,  6)), p1512);
	muac63((vu(19, 16), wp( 9,  8)), p1916);
	muac63((vu(23, 20), wp(11, 10)), p2320);
	muac63((vu(27, 24), wp(13, 12)), p2724);
	muac63((vu(31, 28), wp(15, 14)), p3128);

	muac63((vu( 3,	0), wn( 1,  0)), n0300);
	muac63((vu( 7,	4), wn( 3,  2)), n0704);
	muac63((vu(11,	8), wn( 5,  4)), n1108);
	muac63((vu(15, 12), wn( 7,  6)), n1512);
	muac63((vu(19, 16), wn( 9,  8)), n1916);
	muac63((vu(23, 20), wn(11, 10)), n2320);
	muac63((vu(27, 24), wn(13, 12)), n2724);
	muac63((vu(31, 28), wn(15, 14)), n3128);

	uint4_t p0, p1, p2, p3;
	uint4_t n0, n1, n2, n3;

	uac64((p0300, p0704), p0);
	uac64((p1108, p1512), p1);
	uac64((p1916, p2320), p2);
	uac64((p2724, p3128), p3);

	uac64((n0300, n0704), n0);
	uac64((n1108, n1512), n1);
	uac64((n1916, n2320), n2);
	uac64((n2724, n3128), n3);

	return ((p0 + p1) + (p2 + p3)) - ((n0 + n1) + (n2 + n3));
}

template <typename IT, int FL, int CL, int K>
class Dense {
public:
	void compute(fifo<IT>& ins, fifo<int16_t>& outs) {
		static int1x16_t matp[CL * FL / K] = {
0x0000, 0x0000, 0x0244, 0x0444, 0x9020, 0x00a0, 0x0000, 0x0100,
0x1020, 0x0191, 0x0000, 0x0088, 0x4000, 0x4000, 0x0000, 0x0100,
0x0000, 0x8404, 0x0040, 0x0111, 0x0000, 0x23c0, 0x4000, 0x0009,
0x1090, 0x0404, 0x06c0, 0x8c06, 0x4000, 0x4000, 0xc000, 0x0010,
0x0000, 0x0000, 0x10d9, 0x4400, 0x0400, 0x1c06, 0x4008, 0x8804,
0x1023, 0x0000, 0x0440, 0x0400, 0x1021, 0x0100, 0x30a0, 0x0640,
0x0000, 0x9020, 0x0200, 0x1880, 0x0400, 0x041c, 0x9824, 0x0200,
0x0920, 0x0600, 0x8000, 0x0100, 0x0000, 0x23e0, 0x1080, 0x1001,
0x0980, 0x0600, 0x0900, 0x4000, 0x0004, 0x0000, 0x5019, 0x0000,
0x0090, 0x0000, 0x0151, 0x0400, 0x8d0e, 0x4040, 0x0000, 0x2000,
0xb021, 0x0000, 0x3803, 0x0000, 0x0004, 0x0040, 0x10a0, 0x0000,
0x1008, 0x0000, 0x8240, 0x1030, 0x0003, 0x0010, 0x0000, 0x0440,
0xb000, 0x0a80, 0x3001, 0x0000, 0x0880, 0x0150, 0x8004, 0x0440,
0x0000, 0x0400, 0x0008, 0x1012, 0x0240, 0x1000, 0x0011, 0x0008,
0x0980, 0x0000, 0x4000, 0x0000, 0x200d, 0x0000, 0x0400, 0x0200,
0x2001, 0x0800, 0x0000, 0x8546, 0x0200, 0x8544, 0x0000, 0x0000,
0x3020, 0x0200, 0x0904, 0x1030, 0x0000, 0x0402, 0x0408, 0x0002,
0x8000, 0x3828, 0xa004, 0x0480, 0x0006, 0x0140, 0x4000, 0x1402,
0x0020, 0x2800, 0x0004, 0x0040, 0x2004, 0x0090, 0x0010, 0x0000,
0x8d04, 0x0001, 0x0040, 0x0008, 0x0004, 0x0440, 0x2021, 0x1008,
		};
		static int1x16_t matn[CL * FL / K] = {
0x4000, 0x09e0, 0xb0a2, 0xb8ab, 0x4004, 0x0040, 0x4504, 0x0200,
0x0600, 0x0240, 0x8240, 0x0440, 0x280b, 0x12a0, 0x4004, 0x0000,
0x4404, 0x1238, 0x0400, 0x0240, 0x04c8, 0x4004, 0x0900, 0x0000,
0x4444, 0x9028, 0xc100, 0x0010, 0x0280, 0x0690, 0x0608, 0xcc00,
0x0800, 0x0480, 0xc404, 0x30b9, 0x9115, 0x4000, 0x0000, 0x1010,
0x4440, 0x147b, 0x1020, 0x38a1, 0x0640, 0x0640, 0x4444, 0xb820,
0x0202, 0x0400, 0xa504, 0x0740, 0x0020, 0x0240, 0x0410, 0x0000,
0x4400, 0xa100, 0x0a00, 0x0081, 0x8987, 0x4404, 0x8004, 0x0448,
0x0600, 0x30b9, 0x1000, 0x8306, 0x1280, 0x0280, 0x0b42, 0x1da3,
0x4004, 0x0082, 0x4000, 0x3039, 0x0250, 0x0400, 0x8080, 0x0c48,
0x4404, 0x0420, 0x0000, 0x1029, 0x2c80, 0x1028, 0x0600, 0x3082,
0x8344, 0x3c03, 0x4012, 0xc400, 0x0240, 0x1929, 0x30d0, 0x1000,
0x0400, 0x0100, 0x0544, 0x3281, 0x9220, 0x4c0c, 0x0000, 0x0000,
0x1400, 0x0000, 0x1200, 0xc004, 0x1d9b, 0x0500, 0xc240, 0x01a1,
0x1000, 0x0000, 0x2488, 0x0000, 0x0040, 0x040a, 0x103b, 0x8d0e,
0x0c00, 0x8500, 0x0400, 0x0010, 0xa187, 0x0000, 0xa6c2, 0x8144,
0x0600, 0x3124, 0x0090, 0x0004, 0x1801, 0x1220, 0x8144, 0x0000,
0x3eaa, 0xc504, 0x0c00, 0x8104, 0x1020, 0x0400, 0x3039, 0x0040,
0x0906, 0x0040, 0x36d9, 0xc404, 0x0800, 0x4104, 0xa008, 0x0021,
0x1219, 0x0000, 0x0006, 0x0040, 0x3508, 0xa826, 0x0000, 0x4404,
		};
#pragma HLS array_partition variable=matp cyclic factor=CL
#pragma HLS array_partition variable=matn cyclic factor=CL

		for (int j = 0; j < FL / K; j++) {
#pragma HLS pipeline
			IT vu = ins.read();
			for (int i = 0; i < CL; i++) {
#pragma HLS unroll
				int1x16_t wp = matp[j * CL + i];
				int1x16_t wn = matn[j * CL + i];
				int16_t acc = muluadd32(vu, wp, wn);
				outs.write(acc);
			}
		}
	}
};

using MatMul0 = Dense<int2x16_t, 256, 10, 16>;

template <int FL, int K>
void read_input(const int in[FL], fifo<int2x16_t>& ins) {
	for (int j = 0; j < FL / K; j++) {
#pragma HLS pipeline
		int2x16_t val;
		for (int k = 0; k < K; k++) {
#pragma HLS unroll
			uint2_t v = in[j * K + k];
			bit::setu<K>(val, k, v);
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
			int16_t val = outs.read();
			acc[i] += val;
			if (j == FL / K - 1) {
				out[i] = acc[i];
			}
		}
	}
}

void kernel(int in[FLATTEN], int weight[CLASS * FLATTEN], int out[CLASS]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=out

	fifo<int2x16_t> ins("input_fifo");
	fifo<int16_t> outs("output_fifo");

	MatMul0 matmul0;

#pragma HLS dataflow
	read_input<256, 16>(in, ins);
	matmul0.compute(ins, outs);
	write_result<256, 10, 16>(out, outs);
}
