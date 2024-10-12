#include "kernel.hpp"
#include <stdio.h>

const int DEGREE = 128;

using chunk_t = float[DEGREE];

void accum01(chunk_t t) {
	t[1] = t[0] + t[1];
	t[3] = t[2] + t[3];
	t[5] = t[4] + t[5];
	t[7] = t[6] + t[7];
	t[9] = t[8] + t[9];
	t[11] = t[10] + t[11];
	t[13] = t[12] + t[13];
	t[15] = t[14] + t[15];
	t[17] = t[16] + t[17];
	t[19] = t[18] + t[19];
	t[21] = t[20] + t[21];
	t[23] = t[22] + t[23];
	t[25] = t[24] + t[25];
	t[27] = t[26] + t[27];
	t[29] = t[28] + t[29];
	t[31] = t[30] + t[31];
	t[33] = t[32] + t[33];
	t[35] = t[34] + t[35];
	t[37] = t[36] + t[37];
	t[39] = t[38] + t[39];
	t[41] = t[40] + t[41];
	t[43] = t[42] + t[43];
	t[45] = t[44] + t[45];
	t[47] = t[46] + t[47];
	t[49] = t[48] + t[49];
	t[51] = t[50] + t[51];
	t[53] = t[52] + t[53];
	t[55] = t[54] + t[55];
	t[57] = t[56] + t[57];
	t[59] = t[58] + t[59];
	t[61] = t[60] + t[61];
	t[63] = t[62] + t[63];
	t[65] = t[64] + t[65];
	t[67] = t[66] + t[67];
	t[69] = t[68] + t[69];
	t[71] = t[70] + t[71];
	t[73] = t[72] + t[73];
	t[75] = t[74] + t[75];
	t[77] = t[76] + t[77];
	t[79] = t[78] + t[79];
	t[81] = t[80] + t[81];
	t[83] = t[82] + t[83];
	t[85] = t[84] + t[85];
	t[87] = t[86] + t[87];
	t[89] = t[88] + t[89];
	t[91] = t[90] + t[91];
	t[93] = t[92] + t[93];
	t[95] = t[94] + t[95];
	t[97] = t[96] + t[97];
	t[99] = t[98] + t[99];
	t[101] = t[100] + t[101];
	t[103] = t[102] + t[103];
	t[105] = t[104] + t[105];
	t[107] = t[106] + t[107];
	t[109] = t[108] + t[109];
	t[111] = t[110] + t[111];
	t[113] = t[112] + t[113];
	t[115] = t[114] + t[115];
	t[117] = t[116] + t[117];
	t[119] = t[118] + t[119];
	t[121] = t[120] + t[121];
	t[123] = t[122] + t[123];
	t[125] = t[124] + t[125];
	t[127] = t[126] + t[127];
}

void accum02(chunk_t t) {
	t[2] = t[1] + t[3];
	t[6] = t[5] + t[7];
	t[10] = t[9] + t[11];
	t[14] = t[13] + t[15];
	t[18] = t[17] + t[19];
	t[22] = t[21] + t[23];
	t[26] = t[25] + t[27];
	t[30] = t[29] + t[31];
	t[34] = t[33] + t[35];
	t[38] = t[37] + t[39];
	t[42] = t[41] + t[43];
	t[46] = t[45] + t[47];
	t[50] = t[49] + t[51];
	t[54] = t[53] + t[55];
	t[58] = t[57] + t[59];
	t[62] = t[61] + t[63];
	t[66] = t[65] + t[67];
	t[70] = t[69] + t[71];
	t[74] = t[73] + t[75];
	t[78] = t[77] + t[79];
	t[82] = t[81] + t[83];
	t[86] = t[85] + t[87];
	t[90] = t[89] + t[91];
	t[94] = t[93] + t[95];
	t[98] = t[97] + t[99];
	t[102] = t[101] + t[103];
	t[106] = t[105] + t[107];
	t[110] = t[109] + t[111];
	t[114] = t[113] + t[115];
	t[118] = t[117] + t[119];
	t[122] = t[121] + t[123];
	t[126] = t[125] + t[127];
}

void accum03(chunk_t t) {
	t[4] = t[2] + t[6];
	t[12] = t[10] + t[14];
	t[20] = t[18] + t[22];
	t[28] = t[26] + t[30];
	t[36] = t[34] + t[38];
	t[44] = t[42] + t[46];
	t[52] = t[50] + t[54];
	t[60] = t[58] + t[62];
	t[68] = t[66] + t[70];
	t[76] = t[74] + t[78];
	t[84] = t[82] + t[86];
	t[92] = t[90] + t[94];
	t[100] = t[98] + t[102];
	t[108] = t[106] + t[110];
	t[116] = t[114] + t[118];
	t[124] = t[122] + t[126];
}

void accum04(chunk_t t) {
	t[8] = t[4] + t[12];
	t[24] = t[20] + t[28];
	t[40] = t[36] + t[44];
	t[56] = t[52] + t[60];
	t[72] = t[68] + t[76];
	t[88] = t[84] + t[92];
	t[104] = t[100] + t[108];
	t[120] = t[116] + t[124];
}

void accum05(chunk_t t) {
	t[16] = t[8] + t[24];
	t[48] = t[40] + t[56];
	t[80] = t[72] + t[88];
	t[112] = t[104] + t[120];
}

void accum06(chunk_t t) {
	t[32] = t[16] + t[48];
	t[96] = t[80] + t[112];
}

void accum07(chunk_t t) {
	t[64] = t[32] + t[96];
}

float quick_sum(chunk_t buf) {
#pragma HLS inline
	accum01(buf);
	accum02(buf);
	accum03(buf);
	accum04(buf);
	accum05(buf);
	accum06(buf);
	accum07(buf);
	return buf[DEGREE / 2];
}

void kernel(const float in[1024], const int size, float *out) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=size
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE

	float acc[8];
#pragma HLS array_partition variable=acc

	for (int i = 0; i < 1024 / DEGREE; i++) {
#pragma HLS pipeline
		chunk_t buf;
#pragma HLS array_partition variable=buf
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			buf[j] = (i * DEGREE + j < size)
				? in[i * DEGREE + j] : 0;
		}
		acc[i] = quick_sum(buf);
	}

	*out = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
		+ ((acc[4] + acc[5]) + (acc[6] + acc[7]));
}
