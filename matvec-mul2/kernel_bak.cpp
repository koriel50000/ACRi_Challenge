#include "kernel.hpp"

const int CHUNK = SIZE * 4;

void quick_sum(float v[CHUNK / 2]) {
#pragma HLS inline
	float t[CHUNK / 4];
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
	t[60] = v[120] + v[121];
	t[61] = v[122] + v[123];
	t[62] = v[124] + v[125];
	t[63] = v[126] + v[127];
	t[64] = v[128] + v[129];
	t[65] = v[130] + v[131];
	t[66] = v[132] + v[133];
	t[67] = v[134] + v[135];
	t[68] = v[136] + v[137];
	t[69] = v[138] + v[139];
	t[70] = v[140] + v[141];
	t[71] = v[142] + v[143];
	t[72] = v[144] + v[145];
	t[73] = v[146] + v[147];
	t[74] = v[148] + v[149];
	t[75] = v[150] + v[151];
	t[76] = v[152] + v[153];
	t[77] = v[154] + v[155];
	t[78] = v[156] + v[157];
	t[79] = v[158] + v[159];

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
	v[20] = t[40] + t[41];
	v[21] = t[42] + t[43];
	v[22] = t[44] + t[45];
	v[23] = t[46] + t[47];
	v[24] = t[48] + t[49];
	v[25] = t[50] + t[51];
	v[26] = t[52] + t[53];
	v[27] = t[54] + t[55];
	v[28] = t[56] + t[57];
	v[29] = t[58] + t[59];
	v[30] = t[60] + t[61];
	v[31] = t[62] + t[63];
	v[32] = t[64] + t[65];
	v[33] = t[66] + t[67];
	v[34] = t[68] + t[69];
	v[35] = t[70] + t[71];
	v[36] = t[72] + t[73];
	v[37] = t[74] + t[75];
	v[38] = t[76] + t[77];
	v[39] = t[78] + t[79];

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

	v[0] = t[0] + t[1];
	v[1] = t[2] + t[3];
	v[5] = t[5] + t[6];
	v[6] = t[7] + t[8];
	v[10] = t[10] + t[11];
	v[11] = t[12] + t[13];
	v[15] = t[15] + t[16];
	v[16] = t[17] + t[18];

	t[0] = v[0] + v[1];
	t[5] = v[5] + v[6];
	t[10] = v[10] + v[11];
	t[15] = v[15] + v[16];

	v[0] = t[0] + t[4];
	v[5] = t[5] + t[9];
	v[10] = t[10] + t[14];
	v[15] = t[15] + t[19];
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

	float vec[CHUNK];
#pragma HLS array_partition variable=vec

	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		float v = in_vec[i];
		vec[SIZE * 0 + i] = v;
		vec[SIZE * 1 + i] = v;
		vec[SIZE * 2 + i] = v;
		vec[SIZE * 3 + i] = v;
	}

	for (int i = 0, j = 0; j < SIZE; i += CHUNK, j += 4) {
#pragma HLS pipeline
		float val[CHUNK];
#pragma HLS array_partition variable=val

		for (int k = 0; k < CHUNK; k += 2) {
#pragma HLS unroll
			float v0 = in_mat[i + k + 0] * vec[k + 0];
			float v1 = in_mat[i + k + 1] * vec[k + 1];
			val[k / 2] = v0 + v1;
		}

		quick_sum(val);
		out[j + 0] = val[0];
		out[j + 1] = val[5];
		out[j + 2] = val[10];
		out[j + 3] = val[15];
	}
}
