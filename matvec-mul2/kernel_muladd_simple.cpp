#include "kernel.hpp"

const int HALF = SIZE / 2;
const int CHUNK = SIZE + HALF;

template <typename T, int M>
T muladd(T mat[M], T vec[M]) {
	T t[M];
#pragma HLS array_partition variable=t

	t[0] = mat[0] * vec[0];
	t[1] = mat[1] * vec[1];
	t[2] = mat[2] * vec[2];
	t[3] = mat[3] * vec[3];
	t[4] = mat[4] * vec[4];
	t[5] = mat[5] * vec[5];
	t[6] = mat[6] * vec[6];
	t[7] = mat[7] * vec[7];
	t[8] = mat[8] * vec[8];
	t[9] = mat[9] * vec[9];
	t[10] = mat[10] * vec[10];
	t[11] = mat[11] * vec[11];
	t[12] = mat[12] * vec[12];
	t[13] = mat[13] * vec[13];
	t[14] = mat[14] * vec[14];
	t[15] = mat[15] * vec[15];
	t[16] = mat[16] * vec[16];
	t[17] = mat[17] * vec[17];
	t[18] = mat[18] * vec[18];
	t[19] = mat[19] * vec[19];
	t[20] = mat[20] * vec[20];
	t[21] = mat[21] * vec[21];
	t[22] = mat[22] * vec[22];
	t[23] = mat[23] * vec[23];
	t[24] = mat[24] * vec[24];
	t[25] = mat[25] * vec[25];
	t[26] = mat[26] * vec[26];
	t[27] = mat[27] * vec[27];
	t[28] = mat[28] * vec[28];
	t[29] = mat[29] * vec[29];
	t[30] = mat[30] * vec[30];
	t[31] = mat[31] * vec[31];
	t[32] = mat[32] * vec[32];
	t[33] = mat[33] * vec[33];
	t[34] = mat[34] * vec[34];
	t[35] = mat[35] * vec[35];
	t[36] = mat[36] * vec[36];
	t[37] = mat[37] * vec[37];
	t[38] = mat[38] * vec[38];
	t[39] = mat[39] * vec[39];
	t[40] = mat[40] * vec[40];
	t[41] = mat[41] * vec[41];
	t[42] = mat[42] * vec[42];
	t[43] = mat[43] * vec[43];
	t[44] = mat[44] * vec[44];
	t[45] = mat[45] * vec[45];
	t[46] = mat[46] * vec[46];
	t[47] = mat[47] * vec[47];
	t[48] = mat[48] * vec[48];
	t[49] = mat[49] * vec[49];
	t[50] = mat[50] * vec[50];
	t[51] = mat[51] * vec[51];
	t[52] = mat[52] * vec[52];
	t[53] = mat[53] * vec[53];
	t[54] = mat[54] * vec[54];
	t[55] = mat[55] * vec[55];
	t[56] = mat[56] * vec[56];
	t[57] = mat[57] * vec[57];
	t[58] = mat[58] * vec[58];
	t[59] = mat[59] * vec[59];
	t[60] = mat[60] * vec[60];
	t[61] = mat[61] * vec[61];
	t[62] = mat[62] * vec[62];
	t[63] = mat[63] * vec[63];
	t[64] = mat[64] * vec[64];
	t[65] = mat[65] * vec[65];
	t[66] = mat[66] * vec[66];
	t[67] = mat[67] * vec[67];
	t[68] = mat[68] * vec[68];
	t[69] = mat[69] * vec[69];
	t[70] = mat[70] * vec[70];
	t[71] = mat[71] * vec[71];
	t[72] = mat[72] * vec[72];
	t[73] = mat[73] * vec[73];
	t[74] = mat[74] * vec[74];
	t[75] = mat[75] * vec[75];
	t[76] = mat[76] * vec[76];
	t[77] = mat[77] * vec[77];
	t[78] = mat[78] * vec[78];
	t[79] = mat[79] * vec[79];

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
	t[60] += t[61];
	t[62] += t[63];
	t[64] += t[65];
	t[66] += t[67];
	t[68] += t[69];
	t[70] += t[71];
	t[72] += t[73];
	t[74] += t[75];
	t[76] += t[77];
	t[78] += t[79];

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
	t[60] += t[62];
	t[64] += t[66];
	t[68] += t[70];
	t[72] += t[74];
	t[76] += t[78];

	t[0] += t[4];
	t[8] += t[12];
	t[16] += t[20];
	t[24] += t[28];
	t[32] += t[36];
	t[40] += t[44];
	t[48] += t[52];
	t[56] += t[60];
	t[64] += t[68];
	t[72] += t[76];

	t[0] += t[8];
	t[16] += t[24];
	t[32] += t[40];
	t[48] += t[56];
	t[64] += t[72];

	t[0] += t[16];
	t[32] += t[48];

	t[0] += t[32];

	t[0] += t[64];

	return t[0];
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

	float vec[SIZE];
#pragma HLS array_partition variable=vec

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		vec[i] = in_vec[i];
	}

	float mat0[SIZE];
	float mat1[SIZE];
	float mat2[SIZE];
#pragma HLS array_partition variable=mat0
#pragma HLS array_partition variable=mat1
#pragma HLS array_partition variable=mat2

#pragma HLS pipeline
	int ptr = 0;
	for (int i = 0; i < SIZE; i += 3) {
		for (int j = 0; j < CHUNK; j++) {
#pragma HLS unroll
			mat0[j] = in_mat[ptr++];
		}
		out[i + 0] = muladd0<float, SIZE>(mat0, vec);

		for (int j = 0; j < SIZE; j++) {
#pragma HLS unroll
			mat1[j] = in_mat[ptr++];
		}
		out[i + 1] = muladd1<float, SIZE>(mat0, mat1, vec);

		if (i + 2 == SIZE) break;
		out[i + 2] = muladd2<float, SIZE>(mat1, vec);
	}
}
