#include "kernel.hpp"
#include <hls_vector.h>
#include <hls_stream.h>

const int TOP5 = 5;
const int DEGREE = 32;

typedef hls::vector<float, TOP5> top5_t;

typedef hls::vector<float, DEGREE> data_t;
typedef hls::stream<data_t> fifo_t;

void insert_sort(top5_t& v, const float f) {
	float t0 = v[0];
	float t1 = v[1];
	float t2 = v[2];
	float t3 = v[3];
	float t4 = v[4];
	bool b0 = (f > v[0]);
	bool b1 = (f > v[1]);
	bool b2 = (f > v[2]);
	bool b3 = (f > v[3]);
	bool b4 = (f > v[4]);
	v[0] = b0 ? f : t0;
	v[1] = b0 ? t0 : (b1 ? f : t1);
	v[2] = b1 ? t1 : (b2 ? f : t2);
	v[3] = b2 ? t2 : (b3 ? f : t3);
	v[4] = b3 ? t3 : (b4 ? f : t4);
}

float lshift(top5_t& v) {
	float t = v[0];
	v[0] = v[1];
	v[1] = v[2];
	v[2] = v[3];
	v[3] = v[4];
	return t;
}

void read_input(const float in[SIZE], fifo_t& outs) {
	static top5_t buf[DEGREE];
#pragma HLS array_partition variable=buf

	for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
		buf[j] = -2000.0;
	}

	for (int i = 0; i < SIZE; i += DEGREE) {
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			insert_sort(buf[j], in[i + j]);
		}
	}

	for (int i = 0; i < TOP5; i++) {
		data_t data;
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			data[j] = lshift(buf[j]);
		}
		outs << data;
	}
}

void write_result(float out[TOP5], fifo_t& outs) {
	static top5_t top = -2000.0;

	for (int i = 0; i < TOP5; i++) {
#pragma HLS pipeline
		data_t data;
		outs >> data;
		for (int j = 0; j < DEGREE; j++) {
			insert_sort(top, data[j]);
		}
		out[i] = lshift(top);
	}
}

void kernel(const float in[SIZE], float out[5]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE

	fifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input(in, outs);
	write_result(out, outs);
}
