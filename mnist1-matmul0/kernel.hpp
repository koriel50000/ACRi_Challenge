#pragma once
#include <ap_int.h>
#include <hls_stream.h>

const int FLATTEN = 256;
const int CLASS = 10;

const int CHUNK_SIZE = 16;

using uint2_t = ap_uint<2>;
using uint3_t = ap_uint<3>;
using uint4_t = ap_uint<4>;
using uint6_t = ap_uint<6>;

template <typename T>
using fifo = hls::stream<T>;

class int_t {
private:
	ap_uint<W* N> buf_;
public:
	int_t();
	int_t(int i);
	int_t(unsigned int ui);
	int_t(long l);
	int_t(unsigned long ul);
	int_t(const char* s);

	ap_range_ref<W* N, false> operator[](size_t index) const;
	ap_range_ref<W* N, false> operator[](size_t index);
};

extern "C" {
void kernel(
  int in[256],
  int weight[10 * 256],
  int out[10]
);
}
