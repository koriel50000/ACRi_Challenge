#include "kernel.hpp"
#include <hls_stream.h>
#include <hls_vector.h>
#include <ap_int.h>

const int DEGREE = 32;
const int BIT_WIDTH = 5;

typedef ap_uint<BIT_WIDTH> char_t;
typedef ap_uint<PATTERN_SIZE> mask_t;
typedef ap_uint<DEGREE> match_t;

typedef hls::vector<char_t, PATTERN_SIZE + DEGREE> buffer_t;
typedef hls::vector<char_t, PATTERN_SIZE> word_t;

typedef struct {
	bool end;
	int pos;
	match_t match;
} matcher_t;

typedef hls::stream<buffer_t> ififo_t;
typedef hls::stream<matcher_t> ofifo_t;

const char_t UNDEFINED = 0x1f;

char_t to_char(char ch) {
#pragma HLS inline
	return ch - '`';
}

void shift_buf(buffer_t& buf) {
	for (int j = 0; j < PATTERN_SIZE; j++) {
#pragma HLS unroll
		buf[j] = buf[DEGREE + j];
	}
}
void range_buf(const buffer_t& buf, const int offset, word_t& word) {
	for (int i = 0; i < PATTERN_SIZE; i++) {
#pragma HLS unroll
		word[i] = buf[offset + i];
	}
}

// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/coQM3TlQjohHv6MKC6dg
bool compare_string(const word_t& str, const word_t& pat, const mask_t& mask) {
	mask_t notmatch;
        for (int i = 0; i < PATTERN_SIZE; i++) {
#pragma HLS unroll
                notmatch[i] = (str[i] != pat[i]) ? 1 : 0;
        }
        return ((notmatch & mask) == 0);
}

void input_text(const char in_text[TEXT_SIZE], ififo_t& ins) {
	buffer_t buf;

	for (int i = 0; i < PATTERN_SIZE; i++) {
#pragma HLS unroll
		buf[i] = to_char(in_text[i]);
	}

	for (int i = PATTERN_SIZE; i < TEXT_SIZE + PATTERN_SIZE; i += DEGREE) {
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			buf[PATTERN_SIZE + j] = (i + j < TEXT_SIZE) ? to_char(in_text[i + j]) : UNDEFINED;
		}
		ins << buf;
		shift_buf(buf);
	}
}

void search(const char in_pat[PATTERN_SIZE], ififo_t& ins, ofifo_t& outs) {
	word_t pat;
	mask_t mask;

	for (int i = 0; i < PATTERN_SIZE; i++) {
#pragma HLS unroll
		char ch = in_pat[i];
		pat[i] = to_char(ch);
		mask[i] = (ch != '?') ? 1 : 0;
	}

	matcher_t matcher;
	matcher.end = false;
	for (matcher.pos = 0; matcher.pos < TEXT_SIZE; matcher.pos += DEGREE) {
		buffer_t buf;
		ins >> buf;

		// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/coQM3TlQjohHv6MKC6dg
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			word_t str;
			range_buf(buf, j, str);
			matcher.match[j] = compare_string(str, pat, mask);
		}
		if (matcher.match != 0) {
			outs << matcher;
		}
	}
	matcher.end = true;
	outs << matcher;
}

void output_pos(int out_match_pos[MAX_MATCH], int* out_match_num, ofifo_t& outs) {
	int count = 0;

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
	input_text(in_text, ins);
	search(in_pat, ins, outs);
	output_pos(out_match_pos, out_match_num, outs);
}
