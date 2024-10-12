#include "kernel.hpp"
#include <ap_int.h>

const int CHUNK_SIZE = 32;

typedef char buffer_t[PATTERN_SIZE + CHUNK_SIZE];
typedef char block_t[PATTERN_SIZE];
typedef ap_uint<PATTERN_SIZE> mask_t;

typedef ap_uint<CHUNK_SIZE> matcher_t;

void range_buf(const buffer_t buf, const int offset, block_t str) {
	for (int j = 0; j < PATTERN_SIZE; j++) {
#pragma HLS unroll
		str[j] = buf[j + offset];
	}
}

bool compare_string(const block_t str, const block_t pat, const mask_t mask) {
	mask_t match;

	for (int j = 0; j < PATTERN_SIZE; j++) {
#pragma HLS unroll
		// FIXME このif文に問題がある？
		// if (pat[j] != '?' && pat[j] != buf[j]) {
		match[j] = (pat[j] != str[j]) ? 1 : 0;
	}
	return (match & mask) == 0;
}

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

	block_t pat;
	mask_t mask;
#pragma HLS array_partition variable=pat

        // @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/coQM3TlQjohHv6MKC6dg
	for (int j = 0; j < PATTERN_SIZE; j++) {
#pragma HLS unroll
		char ch = in_pat[j];
		pat[j] = ch;
		mask[j] = (ch != '?') ? 1 : 0;
	}

	buffer_t buf;
#pragma HLS array_partition variable=buf

	for (int i = 0; i < PATTERN_SIZE; i++) {
		buf[i] = in_text[i];
	}

	int match_count = 0;
	matcher_t matcher;

	for (int i = 0; i < TEXT_SIZE; i += CHUNK_SIZE) {
#pragma HLS pipeline
		for (int k = 0; k < CHUNK_SIZE; k++) {
#pragma HLS unroll
			buf[PATTERN_SIZE + k] = PATTERN_SIZE + i < TEXT_SIZE
				? in_text[PATTERN_SIZE + i + k] : 0;
		}

		for (int k = 0; k < CHUNK_SIZE; k++) {
#pragma HLS unroll
			block_t str;
			range_buf(buf, k, str);
			matcher[k] = compare_string(str, pat, mask);
		}

		for (int k = 0; k < CHUNK_SIZE; k++) {
#pragma HLS pipeline
			if (matcher[k] && match_count < MAX_MATCH) {
				out_match_pos[match_count++] = i + k;
			}
		}

		for (int k = 0; k < CHUNK_SIZE; k++) {
#pragma HLS unroll
			buf[k] = buf[k + CHUNK_SIZE];
		}
	}

	*out_match_num = match_count;
}
