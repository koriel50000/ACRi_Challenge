#include "kernel.hpp"
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>

const int DEGREE = 128;

using chunk_t = hls::vector<float, DEGREE>;
template <typename T>
using fifo = hls::stream<T>;

// N = 2, 4, 8, 16, ...
template <typename T, int N>
T quick_sum(const hls::vector<T, N>& v) {
#pragma HLS inline
	const int p = ilogb(N / 2);
	T t[N / 2];
#pragma HLS array_partition variable=t

	for (int i = 0; i < N; i += 2) {
#pragma HLS unroll
		t[i / 2] = v[i] + v[i + 1];
	}
	for (int j = 0, d = 1; j < p; j++, d *= 2) {
		for (int i = 0; i < N / 2; i += d * 2) {
#pragma HLS unroll
			t[i] += t[i + d];
		}
	}
	return t[0];
}

// N = 2, 4, 8, 16, ...
template <typename T, int N>
T quick_expsum(const hls::vector<T, N>& v) {
#pragma HLS inline
	const int p = ilogb(N);
	T t[N];
#pragma HLS array_partition variable=t

	for (int i = 0; i < N; i++) {
#pragma HLS unroll
		t[i] = v[i] * v[i];
	}
	for (int j = 0, d = 1; j < p; j++, d *= 2) {
		for (int i = 0; i < N; i += d * 2) {
#pragma HLS unroll
			t[i] += t[i + d];
		}
	}
	return t[0];
}

void read_input(const float in[1024], const int size,
		fifo<chunk_t>& insm, fifo<chunk_t>& insv)
{
	for (int i = 0; i < size; i += DEGREE) {
#pragma HLS pipeline
		chunk_t buf;
		for (int j = 0; j < DEGREE; j++) {
#pragma HLS unroll
			// @thanks https://acri-vhls-challenge.web.app/user/fpga/code/M8sNwWiPptiEnUn8GdOu
			buf[j] = (i + j < size) ? in[i + j] : 0;
		}
		insm.write(buf);
		insv.write(buf);
	}
}

void compute_expacc(const int count, fifo<chunk_t>& ins, fifo<float>& outs) {
	for (int i = 0; i < count; i++) {
#pragma HLS pipeline
		chunk_t buf = ins.read();
		float acc = quick_expsum<float, DEGREE>(buf);
		outs.write(acc);
	}
}

void compute_accum(const int count, fifo<chunk_t>& ins, fifo<float>& outs) {
	for (int i = 0; i < count; i++) {
#pragma HLS pipeline
		chunk_t buf = ins.read();
		float acc = quick_sum<float, DEGREE>(buf);
		outs.write(acc);
	}
}

void write_result(const int count, float& out, fifo<float>& outs) {
	float acc = 0;
	for (int i = 0; i < count; i++) {
#pragma HLS pipeline
		acc += outs.read();
	}
	out = acc;
}

void vector_mean_vari(const float in[1024], const int size,
		float& sum, float& exp)
{
	fifo<chunk_t> insm("input_mean_fifo");
	fifo<chunk_t> insv("input_vari_fifo");
	fifo<float> outsm("output_mean_fifo");
	fifo<float> outsv("output_vari_fifo");

	const int count = (size + DEGREE - 1) / DEGREE;
#pragma HLS dataflow
	read_input(in, size, insm, insv);
	compute_accum(count, insm, outsm);
	compute_expacc(count, insv, outsv);
	write_result(count, sum, outsm);
	write_result(count, exp, outsv);
}

void kernel(
  const float in[1024],
  const int size,
  float mean[1],
  float vari[1]
) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=size
#pragma HLS interface axis port=mean
#pragma HLS interface axis port=vari
#pragma HLS array_partition variable=in cyclic factor=DEGREE

	float sum, exp;

	vector_mean_vari(in, size, sum, exp);

	// @thanks https://acri-vhls-challenge.web.app/user/nabesan_go/code/rHfUa5w2mfKIapFojgv6
	mean[0] = sum / size;
	vari[0] = (exp - sum * sum / size) / size;
}
