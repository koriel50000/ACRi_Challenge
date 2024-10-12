#include <algorithm>
#include <hls_stream.h>

#include "kernel.hpp"

typedef struct {
	uint64_t code;
	uint8_t size;
} code_t;

typedef struct {
	bool end;
	uint64_t buf;
	uint8_t len;
} data_t;

typedef hls::stream<code_t> ififo_t;
typedef hls::stream<data_t> ofifo_t;

void read_input(const uint8_t data[SIZE], const uint64_t code[256], const uint64_t code_size[256], ififo_t& ins) {
	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll factor=8 skip_exit_check
		code_t val;
		uint8_t ch = data[i];
		val.code = code[ch];
		val.size = code_size[ch];
		ins << val;
	}
}

void compute_encode(ififo_t& ins, ofifo_t& outs) {
	data_t data;
	data.end = false;
	uint64_t buf = 0;
	uint8_t len = 0;
	for (int i = 0; i < SIZE; i++) {
#pragma HLS pipeline II=1
		// @thanks https://acri-vhls-challenge.web.app/user/nabesan_go/code/vq7toeQWUJ3YvaVNUOqv
		code_t inval;
		ins >> inval;
		uint64_t cd = inval.code;
		uint8_t sz = inval.size;
		len += sz;
		buf |= cd << (64 - len);
		if (len >= 8) {
			data.buf = buf;
			data.len = len;
			outs << data;
			buf <<= len & 0xfff8;
			len &= len & 0x07;
		}
	}
	data.end = true;
	data.buf = buf;
	data.len = len;
	outs << data;
}

void write_result(uint8_t out[SIZE * 8], ofifo_t& outs) {
	data_t data;
	uint64_t buf;
	uint8_t len;
	int ptr = 0;
	do {
		outs >> data;
		buf = data.buf;
		len = data.len;
		while (len >= 8) {
#pragma HLS pipeline II=1
			out[ptr++] = buf >> 56;
			buf <<= 8;
			len -= 8;
		}
	} while (!data.end);
	if (len > 0) {
		out[ptr++] = buf >> 56;
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
#pragma HLS interface bram port=code
#pragma HLS interface bram port=code_size
#pragma HLS array_partition variable=data cyclic factor=8
#pragma HLS array_partition variable=code
#pragma HLS array_partition variable=code_size

	ififo_t ins("input_fifo");
	ofifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input(data, code, code_size, ins);
	compute_encode(ins, outs);
	write_result(out, outs);
}
