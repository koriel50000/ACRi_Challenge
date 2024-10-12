#include <algorithm>
#include <hls_stream.h>
#include <hls_vector.h>
#include <ap_int.h>

#include "kernel.hpp"

const int CHUNK_SIZE = 8;
const int BIT_WIDTH = 128;

typedef struct {
	uint64_t code;
	uint8_t size;
} code_t;

typedef struct {
	bool end;
	ap_uint<BIT_WIDTH> buf;
	int len;
} data_t;

typedef hls::vector<code_t, CHUNK_SIZE> chunk_t;
typedef hls::stream<chunk_t> ififo_t;
typedef hls::stream<data_t> ofifo_t;

void read_input(const uint8_t data[SIZE], const uint64_t code[256], const uint64_t code_size[256], ififo_t& ins) {
	uint64_t code_table[256];
	uint8_t code_size_table[256];
#pragma HLS array_partition variable=code_table
#pragma HLS array_partition variable=code_size_table

	for (int i = 0; i < 256; i++) {
#pragma HLS unroll factor=16 skip_exit_check
		uint64_t cd = code[i];
		uint8_t sz = code_size[i];
		code_table[i] = cd << (64 - sz);
		code_size_table[i] = sz;
	}

	int ptr = 0;
	for (int i = 0; i < SIZE / CHUNK_SIZE; i++) {
#pragma HLS pipeline
		chunk_t chunk;
		for (int j = 0; j < CHUNK_SIZE; j++) {
#pragma HLS unroll
			uint8_t ch = data[ptr++];
			chunk[j].code = code_table[ch];
			chunk[j].size = code_size_table[ch];
		}
		ins << chunk;
	}
}

void compute_encode(ififo_t& ins, ofifo_t& outs) {
	data_t data;
	data.end = false;
	ap_uint<BIT_WIDTH> buf = 0;
	int len = 0;
	for (int i = 0; i < SIZE / CHUNK_SIZE; i++) {
#pragma HLS pipeline
		chunk_t chunk;
		ins >> chunk;
		for (int j = 0; j < CHUNK_SIZE; j++) {
			buf(BIT_WIDTH - 1, BIT_WIDTH - 64) = chunk[j].code;
			buf.lrotate(chunk[j].size);
			len += chunk[j].size;
			if (len > BIT_WIDTH - 64) {
				data.buf = buf;
				data.len = len;
				outs << data;
				len &= 0x07;
				buf = (len > 0) ? buf(len - 1, 0) : 0;
			}
		}
	}
	data.end = true;
	data.buf = buf;
	data.len = len;
	outs << data;
}

void write_result(uint8_t out[SIZE * 8], ofifo_t& outs) {
	data_t data;
	ap_uint<BIT_WIDTH> buf;
	int len;
	int ptr = 0;
	do {
#pragma HLS pipeline
		outs >> data;
		buf = data.buf;
		len = data.len;
		buf.rrotate(len);
		while (len >= 8) {
			out[ptr++] = buf(BIT_WIDTH - 1, BIT_WIDTH - 8);
			buf.lrotate(8);
			len -= 8;
		}
	} while (!data.end);
	if (len > 0) {
		out[ptr++] = buf(BIT_WIDTH - 1, BIT_WIDTH - 8);
	}
}

void kernel(
  uint8_t data[SIZE],
  uint64_t code[256],
  uint64_t code_size[256],
  uint8_t out[SIZE*8]
) {
#pragma HLS interface axis port=data
#pragma HLS interface axis port=out
#pragma HLS interface axis port=code
#pragma HLS interface axis port=code_size
#pragma HLS array_partition variable=data cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=out cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=code cyclic factor=16
#pragma HLS array_partition variable=code_size cyclic factor=16

	ififo_t ins("input_fifo");
	ofifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input(data, code, code_size, ins);
	compute_encode(ins, outs);
	write_result(out, outs);
}
