#include "kernel.hpp"
#include <hls_stream.h>
#include <hls_vector.h>
#include <ap_int.h>

const int DEGREE = 32;
const int BIT_WIDTH = 5;

typedef ap_uint<BIT_WIDTH> char_t;
typedef char_t buffer_t[PATTERN_SIZE + DEGREE];
typedef char_t word_t[PATTERN_SIZE];

typedef hls::vector<char_t, DEGREE> block_t;
typedef ap_uint<DEGREE> match_t;

// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/coQM3TlQjohHv6MKC6dg
typedef struct {
	bool end;
	int pos;
	match_t match;
} matcher_t;

typedef hls::stream<block_t> ififo_t;
typedef hls::stream<matcher_t> ofifo_t;

const char_t UNDEFINED = 0x1f;

char_t to_char(char ch) {
#pragma HLS inline
	return ch - '`';
}

void shift_buf(buffer_t buf) {
	for (int i = 0; i < PATTERN_SIZE; i++) {
#pragma HLS unroll
		buf[i] = buf[DEGREE + i];
	}
}

bool compare_string(const buffer_t buf, const int offset, const word_t pat) {
	bool match = true;
	for (int i = 0; i < PATTERN_SIZE; i++) {
#pragma HLS unroll
		if (pat[i] != UNDEFINED && buf[offset + i] != pat[i]) {
			match = false;
		}
	}
	return match;
}

void read_input(const char in_text[TEXT_SIZE], ififo_t& ins) {
	block_t chunk;

	read_input:
	for (int i = 0; i < TEXT_SIZE + PATTERN_SIZE; i += DEGREE) {
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			chunk[j] = (i + j < TEXT_SIZE) ? to_char(in_text[i + j]) : UNDEFINED;
		}
		ins << chunk;
	}
}

void compute_match(const char in_pat[PATTERN_SIZE], ififo_t& ins, ofifo_t& outs) {
	word_t pat;
#pragma HLS array_partition variable=pat

	for (int i = 0; i < PATTERN_SIZE; i++) {
#pragma HLS unroll
		char ch = in_pat[i];
		pat[i] = (ch != '?') ? to_char(in_pat[i]) : UNDEFINED;
	}

	buffer_t buf;
#pragma HLS array_partition variable=buf

	block_t chunk;
	for (int i = 0; i < PATTERN_SIZE; i += DEGREE) {
		ins >> chunk;
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			buf[i + j] = chunk[j];
		}
	}

	matcher_t matcher;
	matcher.end = false;
	compute_match:
	for (matcher.pos = 0; matcher.pos < TEXT_SIZE; matcher.pos += DEGREE) {
		ins >> chunk;
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			buf[PATTERN_SIZE + j] = chunk[j];
			matcher.match[j] = compare_string(buf, j, pat);
		}
		if (matcher.match != 0) {
			outs << matcher;
		}
		shift_buf(buf);
	}
	matcher.end = true;
	outs << matcher;
}

void write_result(int out_match_pos[MAX_MATCH], int* out_match_num, ofifo_t& outs) {
	int count = 0;

	write_result:
	while (true) {
		matcher_t matcher;
		outs >> matcher;
		if (matcher.end) break;
		for (int j = 0; j < DEGREE; j++) {
			if (matcher.match[j] && count < MAX_MATCH) {
				out_match_pos[count++] = matcher.pos + j;
			}
		}
	}

	*out_match_num = count;
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
#pragma HLS array_partition variable=in_text cyclic factor=DEGREE
#pragma HLS array_partition variable=in_pat
#pragma HLS array_partition variable=out_match_pos

	ififo_t ins("input_fifo");
	ofifo_t outs("output_fifo");

	// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/coQM3TlQjohHv6MKC6d1
#pragma HLS dataflow
	read_input(in_text, ins);
	compute_match(in_pat, ins, outs);
	write_result(out_match_pos, out_match_num, outs);
}
