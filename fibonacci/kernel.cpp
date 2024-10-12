#include "kernel.hpp"
#include <stdio.h>

template<int N>
struct fibonacci_t {
	static int compute(int a, int b) {
		return fibonacci_t<N - 1>::compute(b, a + b);
	}
};

template <>
struct fibonacci_t<0> {
	static int compute(int a, int b) {
		return b;
	}
};

int fibonacci_loop(const int n) {
	int a = 1, b = 0;
	for (int i = 0; i < n; i++) {
		int t = a;
		a = b;
		b = t + b;
	}
	return b;
}

void kernel(int* out) {
	*out = fibonacci_t<NUM>::compute(1, 0);
}
