#include "kernel.hpp"

#include "hls_vector.h"
#include "hls_math.h"

const int CHUNK_SIZE = 32;
const int DEGREE = 8;

void kernel(const uint8_t in[8192], const int size, uint16_t hist[256]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=hist
#pragma HLS array_partition variable=in cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=hist cyclic factor=DEGREE

	static uint16_t buf[256 * CHUNK_SIZE];
#pragma HLS array_partition variable=buf cyclic factor=CHUNK_SIZE

	for (int i = 0; i < 256 * CHUNK_SIZE; i++) {
#pragma HLS unroll
		buf[i] = 0;
	}

	const int p = ilogb(CHUNK_SIZE);
	for (int i = 0; i < size; i++) {
// @thanks https://acri-vhls-challenge.web.app/user/NapoliN/code/55l7K2SXGtrZLkBYMc4N
#pragma HLS unroll factor=CHUNK_SIZE
		buf[(in[i] << p) + (i & (CHUNK_SIZE - 1))]++;
	}

	for (int j = 0, len = 2; j < p; j++, len *= 2) {
#pragma HLS pipeline
		for (int i = 0; i < 256 * CHUNK_SIZE; i += len) {
#pragma HLS unroll factor=DEGREE skip_exit_check
			buf[i] += buf[i + len / 2];
		}
	}

	for (int i = 0; i < 256; i++) {
#pragma HLS pipeline
#pragma HLS unroll factor=DEGREE skip_exit_check
		hist[i] = buf[i * CHUNK_SIZE];
	}
}
