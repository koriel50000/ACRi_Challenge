#include "kernel.hpp"
#include <hls_stream.h>
#include <hls_vector.h>

const int HALF = SIZE / 2;
const int CHUNK = SIZE + HALF;
const int COUNT = (SIZE * SIZE + CHUNK - 1) / CHUNK;

using chunk_t = hls::vector<float, CHUNK>;
template <typename T>
using fifo = hls::stream<T>;

void quick_sum(const chunk_t& v, float& o0, float& o1, float& o2) {
	float t[CHUNK / 2];
#pragma HLS array_partition variable=t

	t[0] = v[0] + v[1];
	t[1] = v[2] + v[3];
	t[2] = v[4] + v[5];
	t[3] = v[6] + v[7];
	t[4] = v[8] + v[9];
	t[5] = v[10] + v[11];
	t[6] = v[12] + v[13];
	t[7] = v[14] + v[15];
	t[8] = v[16] + v[17];
	t[9] = v[18] + v[19];
	t[10] = v[20] + v[21];
	t[11] = v[22] + v[23];
	t[12] = v[24] + v[25];
	t[13] = v[26] + v[27];
	t[14] = v[28] + v[29];
	t[15] = v[30] + v[31];
	t[16] = v[32] + v[33];
	t[17] = v[34] + v[35];
	t[18] = v[36] + v[37];
	t[19] = v[38] + v[39];
	t[20] = v[40] + v[41];
	t[21] = v[42] + v[43];
	t[22] = v[44] + v[45];
	t[23] = v[46] + v[47];
	t[24] = v[48] + v[49];
	t[25] = v[50] + v[51];
	t[26] = v[52] + v[53];
	t[27] = v[54] + v[55];
	t[28] = v[56] + v[57];
	t[29] = v[58] + v[59];
	t[30] = v[60] + v[61];
	t[31] = v[62] + v[63];
	t[32] = v[64] + v[65];
	t[33] = v[66] + v[67];
	t[34] = v[68] + v[69];
	t[35] = v[70] + v[71];
	t[36] = v[72] + v[73];
	t[37] = v[74] + v[75];
	t[38] = v[76] + v[77];
	t[39] = v[78] + v[79];
	t[40] = v[80] + v[81];
	t[41] = v[82] + v[83];
	t[42] = v[84] + v[85];
	t[43] = v[86] + v[87];
	t[44] = v[88] + v[89];
	t[45] = v[90] + v[91];
	t[46] = v[92] + v[93];
	t[47] = v[94] + v[95];
	t[48] = v[96] + v[97];
	t[49] = v[98] + v[99];
	t[50] = v[100] + v[101];
	t[51] = v[102] + v[103];
	t[52] = v[104] + v[105];
	t[53] = v[106] + v[107];
	t[54] = v[108] + v[109];
	t[55] = v[110] + v[111];
	t[56] = v[112] + v[113];
	t[57] = v[114] + v[115];
	t[58] = v[116] + v[117];
	t[59] = v[118] + v[119];

	t[0] += t[1];
	t[2] += t[3];
	t[4] += t[5];
	t[6] += t[7];
	t[8] += t[9];
	t[10] += t[11];
	t[12] += t[13];
	t[14] += t[15];
	t[16] += t[17];
	t[18] += t[19];
	t[20] += t[21];
	t[22] += t[23];
	t[24] += t[25];
	t[26] += t[27];
	t[28] += t[29];
	t[30] += t[31];
	t[32] += t[33];
	t[34] += t[35];
	t[36] += t[37];
	t[38] += t[39];
	t[40] += t[41];
	t[42] += t[43];
	t[44] += t[45];
	t[46] += t[47];
	t[48] += t[49];
	t[50] += t[51];
	t[52] += t[53];
	t[54] += t[55];
	t[56] += t[57];
	t[58] += t[59];

	t[0] += t[2];
	t[4] += t[6];
	t[8] += t[10];
	t[12] += t[14];
	t[16] += t[18];
	t[20] += t[22];
	t[24] += t[26];
	t[28] += t[30];
	t[32] += t[34];
	t[36] += t[38];
	t[40] += t[42];
	t[44] += t[46];
	t[48] += t[50];
	t[52] += t[54];
	t[56] += t[58];

	t[0] += t[4];
	t[8] += t[12];
	t[20] += t[24];
	t[28] += t[32];
	t[40] += t[44];
	t[48] += t[52];

	t[0] += t[8];
	t[20] += t[28];
	t[40] += t[48];

	o0 = t[0] + t[16];
	o1 = t[20] + t[36];
	o2 = t[40] + t[56];
}

void read_input(const float in_mat[SIZE * SIZE], fifo<chunk_t>& ins) {
	for (int i = 0; i < SIZE * SIZE + CHUNK - 1; i += CHUNK) {
#pragma HLS pipeline
		chunk_t val;
		for (int j = 0; j < CHUNK; j++) {
#pragma HLS unroll
			val[j] = (i + j < SIZE * SIZE) ? in_mat[i + j] : 0;
		}
		ins.write(val);
	}
}

void compute_matmul(const float in_vec[SIZE],
		fifo<chunk_t>& ins, fifo<chunk_t>& outs)
{
	float vec0[CHUNK];
	float vec1[CHUNK];
#pragma HLS array_partition variable=vec0
#pragma HLS array_partition variable=vec1

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		float v = in_vec[i];
		vec0[i] = v;
		vec1[HALF + i] = v;
		if (i < HALF) {
			vec0[SIZE + i] = v;
		} else {
			vec1[i - HALF] = v;
		}
	}

	for (int i = 0; i < COUNT / 2; i++) {
#pragma HLS pipeline
		chunk_t val;
		chunk_t mat0 = ins.read();
		for (int j = 0; j < CHUNK; j += 2) {
#pragma HLS unroll
			float v0 = mat0[j + 0] * vec0[j + 0];
			float v1 = mat0[j + 1] * vec0[j + 1];
			val[j / 2] = v0 + v1;
		}
		chunk_t mat1 = ins.read();
		for (int j = 0; j < CHUNK; j += 2) {
#pragma HLS unroll
			float v0 = mat1[j + 0] * vec1[j + 0];
			float v1 = mat1[j + 1] * vec1[j + 1];
			val[CHUNK / 2 + j / 2] = v0 + v1;
		}
		outs.write(val);
	}
}

void write_result(float out[SIZE], fifo<chunk_t>& outs) {
	for (int i = 0; i < SIZE; i += 3) {
#pragma HLS pipeline
		const chunk_t val = outs.read();
		float o0, o1, o2;
		quick_sum(val, o0, o1, o2);
		out[i + 0] = o0;
		out[i + 1] = o1;
		if (i + 2 == SIZE) break;
		out[i + 2] = o2;
	}
}

void kernel(
  const float in_mat[SIZE * SIZE],
  const float in_vec[SIZE],
  float out[SIZE]
) {
#pragma HLS interface axis port=in_mat
#pragma HLS interface axis port=in_vec
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in_mat cyclic factor=CHUNK
#pragma HLS array_partition variable=in_vec
#pragma HLS array_partition variable=out

	fifo<chunk_t> ins("input_fifo");
	fifo<chunk_t> outs("output_fifo");

#pragma HLS dataflow
	read_input(in_mat, ins);
	compute_matmul(in_vec, ins, outs);
	write_result(out, outs);
}
