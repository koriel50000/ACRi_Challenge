#pragma once

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>

const int WIDTH = 640;
const int HEIGHT = 640;

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

using int4_t = ap_int<4>;
using bit_t = ap_int<1>;

template <typename T, int N>
using win_t = hls::vector<T, N>;

template <typename T>
using fifo = hls::stream<T>;

extern "C" {
void kernel(
  int in[HEIGHT * WIDTH],
  int out[16]
);
}
