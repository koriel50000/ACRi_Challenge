#include "kernel.hpp"

#include "hls_vector.h"
#include "hls_stream.h"
#include "hls_math.h"

const int CHUNK_SIZE = 64;
const int DEGREE = 32;

typedef hls::vector<uni8_t, CHUNK_SIZE> chunk_t;
typedef hls::stream<data_t> ififo_t;
typedef hls::stream<uint16_t> ofifo_t;

void read_input(const uint8_t in[8192], const int size, ififo_t& ins) {
	for (int i = 0; i < size; i += CHUNK_SIZE) {
		for (int k = 0; k < CHUNK_SIZE; k++) {
#pragma HLS unroll
			chunk[k] = in[i + k];
		}
		data.degree = i & (DEGREE - 1);
		ins << data;
	}
}

void compute_sum(ififo_t& ins, ofifo_t& outs) {
	static uint16_t buf[256 * DEGREE];
#pragma HLS array_partition variable=buf cyclic factor=DEGREE

	for (int i = 0; i < 256 * DEGREE; i++) {
#pragma HLS unroll
		buf[i] = 0;
	}
	const int p = ilogb(DEGREE);

	while (true) {
#pragma HLS unroll factor=DEGREE skip_exit_check
		data_t data;
		ins >> data;
		if (data.end) break;
// @thanks https://acri-vhls-challenge.web.app/user/NapoliN/code/55l7K2SXGtrZLkBYMc4N
		buf[(data.value << p) + data.degree]++;
	}

	for (int i = 0; i < 256; i++) {
#pragma HLS pipeline
		outs << quick_sum<uint16_t, DEGREE>(&buf[i * DEGREE]);
	}
}

void write_result(uint16_t hist[256], ofifo_t& outs) {
	for (int i = 0; i < 256; i++) {
#pragma HLS pipeline
		outs >> hist[i];
	}
}

void kernel(const uint8_t in[8192], const int size, uint16_t hist[256]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=size
#pragma HLS interface axis port=hist
#pragma HLS array_partition variable=in cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=hist

	ififo_t ins("input_fifo");
	ofifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input(in, size, ins);
	compute_sum(ins, outs);
	write_result(hist, outs);
}
