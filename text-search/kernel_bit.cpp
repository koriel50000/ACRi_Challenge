#include "kernel.hpp"
#include <hls_stream.h>
#include <ap_int.h>

const int CHUNK_SIZE = 32;

const int BIT_WIDTH = 5;
const char FILL_BITS = 0x1f;

typedef ap_uint<BIT_WIDTH * (PATTERN_SIZE + CHUNK_SIZE)> buffer_t;
typedef ap_uint<BIT_WIDTH * PATTERN_SIZE> block_t;
typedef ap_uint<PATTERN_SIZE> mask_t;

typedef struct {
	bool end;
	int pos;
	ap_uint<CHUNK_SIZE> match;
} matcher_t;

void put_buf(buffer_t& buf, const int index, const char value) {
#pragma HLS inline
	buf((index + 1) * BIT_WIDTH - 1, index * BIT_WIDTH) = value;
}

buffer_t get_buf(const buffer_t& buf, const int index) {
#pragma HLS inline
        return buf((index + 1) * BIT_WIDTH - 1, index * BIT_WIDTH);
}

void shift_buf(buffer_t& buf, int n) {
#pragma HLS inline
	buf >>= BIT_WIDTH * n;
}

block_t range_buf(const buffer_t& buf, const int offset) {
#pragma HLS inline
	return buf((offset + PATTERN_SIZE) * BIT_WIDTH - 1, offset * BIT_WIDTH);
}

void put_block(block_t& block, const int index, const char value) {
#pragma HLS inline
	block((index + 1) * BIT_WIDTH - 1, index * BIT_WIDTH) = value;
}

block_t get_block(const block_t& block, const int index) {
#pragma HLS inline
        return block((index + 1) * BIT_WIDTH - 1, index * BIT_WIDTH);
}

bool zero(const block_t& block) {
        // FIXME zeroテストがないわけない
	ap_uint<1> bit = 0;
        for (int i = 0; i < block.length(); i++) {
#pragma HLS unroll
                bit |= block[i];
        }
        return bit == 0;
}

void input_text(const char in_text[TEXT_SIZE], hls::stream<buffer_t>& in_fifo) {
	buffer_t buf = 0;

	for (int i = 0; i < PATTERN_SIZE; i++) {
#pragma HLS unroll
		put_buf(buf, i, in_text[i]);
	}

	for (int i = PATTERN_SIZE; i < TEXT_SIZE; i += CHUNK_SIZE) {
#pragma HLS pipeline
		for (int k = 0; k < CHUNK_SIZE; k++) {
#pragma HLS unroll
			put_buf(buf, PATTERN_SIZE + k, in_text[i + k]);
		}
		in_fifo << buf;
		shift_buf(buf, CHUNK_SIZE);
	}
	in_fifo << buf;
}

void search(const char in_pat[PATTERN_SIZE], hls::stream<buffer_t>& in_fifo, hls::stream<matcher_t>& out_fifo) {
	block_t pat = 0;
	block_t mask = 0;

	for (int j = 0; j < PATTERN_SIZE; j++) {
#pragma HLS unroll
		char ch = in_pat[j];
		if (ch != '?') {
			put_block(pat, j, ch);
			put_block(mask, j, FILL_BITS);
		}
	}

	for (int i = 0; i < TEXT_SIZE; i+= CHUNK_SIZE) {
#pragma HLS pipeline
		buffer_t buf;
		in_fifo >> buf;

		matcher_t matcher;
		for (int k = 0; k < CHUNK_SIZE; k++) {
#pragma HLS unroll
			block_t str = range_buf(buf, k);
			matcher.match[k] = zero(str & mask ^ pat);
		}
		matcher.pos = i;
		matcher.end = (i + CHUNK_SIZE >= TEXT_SIZE);
		if (matcher.match != 0 || matcher.end) {
			out_fifo << matcher;
		}
	}
}

void output_pos(int out_match_pos[MAX_MATCH], int* out_match_num, hls::stream<matcher_t>& out_fifo) {
	int match_count = 0;
	matcher_t matcher;

	do {
#pragma HLS pipeline
		out_fifo >> matcher;
		for (int k = 0; k < CHUNK_SIZE; k++) {
			if (matcher.match[k] && match_count < MAX_MATCH) {
				out_match_pos[match_count++] = matcher.pos + k;
			}
		}
	} while (!matcher.end);

	*out_match_num = match_count;
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

	static hls::stream<buffer_t> in_fifo("input_fifo");
	static hls::stream<matcher_t> out_fifo("output_fifo");

	// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/coQM3TlQjohHv6MKC6dg
#pragma HLS dataflow
	input_text(in_text, in_fifo);
	search(in_pat, in_fifo, out_fifo);
	output_pos(out_match_pos, out_match_num, out_fifo);
}
