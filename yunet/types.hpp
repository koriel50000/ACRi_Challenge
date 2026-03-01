#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#undef INLINE
#include <ap_int.h>

using uint4_t = ap_uint<4>;

template <int N, int W = 4>
class int_t {
private:
	ap_uint<W*N> buf_;
public:
	int_t() : buf_(0) {}
	int_t(int i) : buf_(i) {}
	int_t(uint64_t ul) : buf_(ul) {}
	int_t(uint64_t w3, uint64_t w2, uint64_t w1, uint64_t w0) {
		assert(N == 64 && W == 4);
		buf_.range(256 - 1, 192) = w3;
		buf_.range(192 - 1, 128) = w2;
		buf_.range(128 - 1,  64) = w1;
		buf_.range( 64 - 1,   0) = w0;
	}

	inline ap_range_ref<W*N, false> operator[](size_t index) const {
		assert(index < N);
		return buf_(W * index + W - 1, W * index);
	}

	inline ap_range_ref<W*N, false> operator[](size_t index) {
		assert(index < N);
		return buf_(W * index + W - 1, W * index);
	}

	unsigned long to_ulong() const {
		return buf_.to_ulong();
	}
};
