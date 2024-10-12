#include "kernel.hpp"

void kernel(
  const char in_text[TEXT_SIZE],
  const char in_pat[PATTERN_SIZE],
  int out_match_pos[MAX_MATCH],
  int* out_match_num
) {
#pragma HLS interface axis port=in_text
#pragma HLS interface axis port=in_pat
#pragma HLS interface axis port=out_match_pos
#pragma HLS array_partition variable=in_text cyclic factor=32
#pragma HLS array_partition variable=in_pat
#pragma HLS array_partition variable=out_match_pos

	char pat[PATTERN_SIZE];
#pragma HLS array_partition variable=pat
	// 末尾の'?'は無視する仕様
	int epos = 0;
	for (int j = 0; j < PATTERN_SIZE; j++) {
#pragma HLS unroll
		pat[j] = in_pat[j];
		if (pat[j] != '?') {
			epos = j;
		}
	}

	static char buf[PATTERN_SIZE] = { '\0' };
#pragma HLS array_partition variable=buf

	int match_count = 0;
	for (int i = 0; i < TEXT_SIZE; i++) {
#pragma HLS pipeline
		buf[0] = in_text[i];
		bool match = true;
		for (int j = epos; j >= 0; --j) {
			if (pat[j] != '?' && pat[j] != buf[epos - j]) {
				match = false;
				break;
			}
		}
		if (match) {
			out_match_pos[match_count++] = i - epos;
			if (match_count >= MAX_MATCH) break;
		}
		for (int j = PATTERN_SIZE - 1; j > 0; --j) {
#pragma HLS unroll
			buf[j] = buf[j - 1];
		}
	}

	*out_match_num = match_count;
}
