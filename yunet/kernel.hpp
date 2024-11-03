#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>

template <int W, int N>
class int_t {
private:
	ap_uint<W*N> buf_;
public:
	int_t() : buf_(0) {}
	int_t(int i) : buf_(i) {}
	int_t(unsigned int ui) : buf_(ui) {}
	int_t(long l) : buf_(l) {}
	int_t(unsigned long ul) : buf_(ul) {}
	int_t(const char* s) : buf_(s) {}

	inline ap_range_ref<W*N, false> operator[](size_t index) const {
		assert(index < N);
		return buf_(W * N - W * index - 1, W * (N - 1) - W * index);
	}

	inline ap_range_ref<W*N, false> operator[](size_t index) {
		assert(index < N);
		return buf_(W * N - W * index - 1, W * (N - 1) - W * index);
	}
};

template <typename T>
using fifo = hls::stream<T>;

template <typename T, int N>
using win_t = hls::vector<T, N>;

using int4_t = ap_uint<4>;

extern "C" {
void kernel(
  int in[640 * 640],
  int out_obj_8[6400 * 1],
  int out_cls_8[6400 * 1],
  int out_bbox_8[6400 * 4],
  int out_kps_8[6400 * 10],
  int out_obj_16[1600 * 1],
  int out_cls_16[1600 * 1],
  int out_bbox_16[1600 * 4],
  int out_kps_16[1600 * 10],
  int out_obj_32[400 * 1],
  int out_cls_32[400 * 1],
  int out_bbox_32[400 * 4],
  int out_kps_32[400 * 10]
);
}
