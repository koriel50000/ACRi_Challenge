#include "kernel.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>

#define I16(i) int_t<1,16>({ (i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1, (i >> 4) & 1, (i >> 5) & 1, (i >> 6) & 1, (i >> 7) & 1, (i >> 8) & 1, (i >> 9) & 1, (i >> 10) & 1, (i >> 11) & 1, (i >> 12) & 1, (i >> 13) & 1, (i >> 14) & 1, (i >> 15) & 1, })

const int FLATTEN = 256;
const int CLASS = 10;

const int CHUNK_SIZE = 16;

using uint2_t = ap_uint<2>;
using uint3_t = ap_uint<3>;
using uint4_t = ap_uint<4>;
using uint6_t = ap_uint<6>;

template <int W, int N>
using int_t = hls::vector<ap_uint<W>, N>;
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
