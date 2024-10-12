#include "kernel.hpp"
#include <hls_vector.h>
#include <hls_math.h>

const int DEGREE = 4;

typedef hls::vector<double, DEGREE> chunk_t;

// N = 2, 4, 8, 16, ...
template <typename T, int N>
T quick_sum(const hls::vector<T, N>& v) {
	const int p = ilogb(N) - 1;
	T t[N / 2];
#pragma HLS array_partition variable=t

	int d = 1;
	for (int i = 0; i < N / 2; i++) {
#pragma HLS unroll
		t[i] = v[i * 2] + v[i * 2 + 1];
	}
	for (int j = 0; j < p; j++) {
		d *= 2;
		for (int i = 0; i < N / 2; i += d) {
#pragma HLS unroll
			t[i] += t[i + d / 2];
		}
	}
	return t[0];
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

	static chunk_t sum = 0.0;
	static chunk_t exp = 0.0;

	for (int i = 0; i < size; i++) {
#pragma HLS unroll factor=DEGREE
		float v = in[i];
		sum[i & (DEGREE - 1)] += v;
		exp[i & (DEGREE - 1)] += v * v;
	}
	float avg = quick_sum<double, DEGREE>(sum) / size;
	// in[i] * in[i]
	// - 2 * in[i] * (sum / size)
	// + (sum / size) * (sum / size)
	float std = quick_sum<double, DEGREE>(exp) / size - avg * avg;

	mean[0] = avg;
	vari[0] = std;
}
