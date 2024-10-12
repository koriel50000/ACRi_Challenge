#include "kernel.hpp"
#include <hls_stream.h>
#include <ap_int.h>

const int AFFINE = 256;
const int CLASS = 10;

const int CHUNK_SIZE = 16;

typedef ap_int<2> int2_t;
typedef ap_uint<2> uint2_t;
typedef ap_uint<CHUNK_SIZE * 4> pack_t;
typedef hls::stream<pack_t> fifo_t;

void sign_extend(pack_t& val, int width, int msb) {
	for (int p = 0; p < CHUNK_SIZE * 4; p += width) {
#pragma HLS unroll
		val[p + msb] = val[p + msb - 1];
	}
}

ap_int<7> quick_sum(pack_t val) {
	sign_extend(val, 4, 3);
	val = (val >> 4 & 0x0f0f0f0f0f0f0f0f) + (val & 0x0f0f0f0f0f0f0f0f);
	sign_extend(val, 8, 4);
	val = (val >> 8 & 0x001f001f001f001f) + (val & 0x001f001f001f001f);
	sign_extend(val, 16, 5);
	val = (val >> 16 & 0x0000003f0000003f) + (val & 0x0000003f0000003f);
	sign_extend(val, 32, 6);
	val = (val >> 32 & 0x000000000000007f) + (val & 0x000000000000007f);
	return val(6, 0);
}

template <int HW, int C>
void read_input(const int in[HW * C], fifo_t& ins) {
	int ptr = 0;
	for (int i = 0; i < HW; i++) {
#pragma HLS pipeline
		pack_t val;
		for (int k = 0; k < C; k++) {
#pragma HLS unroll
			uint2_t v = in[ptr++];
			int p = k * 4;
			val(p + 3, p) = v;
		}
		ins.write(val);
	}
}

template <int HW, int C, int CL>
void compute_matmul(
	const int weight[CL * HW * C],
	fifo_t& ins, fifo_t& outs)
{
	pack_t afilter[CL * HW];
	pack_t xfilter[CL * HW];
	pack_t ifilter[CL * HW];
#pragma HLS array_partition variable=afilter cyclic factor=HW
#pragma HLS array_partition variable=xfilter cyclic factor=HW
#pragma HLS array_partition variable=ifilter cyclic factor=HW

	int ptr = 0;
	for (int i = 0; i < CL * HW; i++) {
#pragma HLS pipeline
		for (int k = 0; k < C; k++) {
#pragma HLS unroll
			int v = weight[ptr++];
			int p = k * 4;
			afilter[i](p + 3, p) = (v != 0) ? 0x7 : 0x0;
			xfilter[i](p + 3, p) = (v < 0) ? 0x7 : 0x0;
			ifilter[i](p + 3, p) = (v < 0) ? 0x1 : 0x0;
		}
	}

	pack_t buf[HW];
#pragma HLS array_partition variable=buf

	for (int j = 0; j < HW; j++) {
#pragma HLS pipeline
		buf[j] = ins.read();
	}

	for (int i = 0; i < CL; i++) {
		for (int j = 0; j < HW; j++) {
#pragma HLS pipeline
			pack_t val = buf[j];
			val ^= xfilter[i * HW + j];
			val += ifilter[i * HW + j];
			val &= afilter[i * HW + j];
			outs.write(val);
		}
	}
}

template <int HW, int C, int CL>
void write_result(int out[CL], fifo_t& outs) {
	for (int i = 0; i < CL; i++) {
#pragma HLS pipeline
		int acc = 0;
		for (int j = 0; j < HW; j++) {
			pack_t val = outs.read();
			acc += quick_sum(val);
		}
		out[i] = acc;
	}
}

void kernel(int in[AFFINE], int weight[CLASS * AFFINE], int out[CLASS]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=weight
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=weight cyclic factor=CHUNK_SIZE

	fifo_t ins("input_fifo");
	fifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input<16, 16>(in, ins);
	compute_matmul<16, 16, 10>(weight, ins, outs);
	write_result<16, 16, 10>(out, outs);
}
