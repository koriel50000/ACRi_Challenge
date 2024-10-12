#include "kernel.hpp"

void quick_sum(float v[SIZE]) {
#pragma HLS inline
	float t[SIZE / 2];
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

	v[0] = t[0] + t[1];
	v[1] = t[2] + t[3];
	v[2] = t[4] + t[5];
	v[3] = t[6] + t[7];
	v[4] = t[8] + t[9];
	v[5] = t[10] + t[11];
	v[6] = t[12] + t[13];
	v[7] = t[14] + t[15];
	v[8] = t[16] + t[17];
	v[9] = t[18] + t[19];
	v[10] = t[20] + t[21];
	v[11] = t[22] + t[23];
	v[12] = t[24] + t[25];
	v[13] = t[26] + t[27];
	v[14] = t[28] + t[29];
	v[15] = t[30] + t[31];
	v[16] = t[32] + t[33];
	v[17] = t[34] + t[35];
	v[18] = t[36] + t[37];
	v[19] = t[38] + t[39];

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

	v[0] = t[0] + t[1];
	v[1] = t[2] + t[3];
	v[2] = t[4] + t[5];
	v[3] = t[6] + t[7];
	v[4] = t[8] + t[9];

	t[0] = v[0] + v[1];
	t[1] = v[2] + v[3];

	v[2] = t[0] + t[1];

	v[0] = v[2] + v[4];
}

void kernel(
  const float in_mat[SIZE * SIZE],
  const float in_vec[SIZE],
  float out[SIZE]
) {
#pragma HLS interface axis port=in_mat
#pragma HLS interface axis port=in_vec
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in_mat cyclic factor=SIZE * 4
#pragma HLS array_partition variable=in_vec
#pragma HLS array_partition variable=out

	float vec[SIZE];
#pragma HLS array_partition variable=vec

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		vec[i] = in_vec[i];
	}

	for (int i = 0, j = 0; j < SIZE; i += SIZE * 4, j += 4) {
#pragma HLS pipeline
		float val0[SIZE];
		float val1[SIZE];
		float val2[SIZE];
		float val3[SIZE];
#pragma HLS array_partition variable=val0
#pragma HLS array_partition variable=val1
#pragma HLS array_partition variable=val2
#pragma HLS array_partition variable=val3

		for (int k = 0; k < SIZE; k++) {
#pragma HLS unroll
			val0[k] = in_mat[i + SIZE * 0 + k] * vec[k];
		}
		quick_sum(val0);
		out[j + 0] = val0[0];

		for (int k = 0; k < SIZE; k++) {
#pragma HLS unroll
			val1[k] = in_mat[i + SIZE * 1 + k] * vec[k];
		}
		quick_sum(val1);
		out[j + 1] = val1[0];

		for (int k = 0; k < SIZE; k++) {
#pragma HLS unroll
			val2[k] = in_mat[i + SIZE * 2 + k] * vec[k];
		}
		quick_sum(val2);
		out[j + 2] = val2[0];

		for (int k = 0; k < SIZE; k++) {
#pragma HLS unroll
			val3[k] = in_mat[i + SIZE * 3 + k] * vec[k];
		}
		quick_sum(val3);
		out[j + 3] = val3[0];
	}
}
