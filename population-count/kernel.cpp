#include "kernel.hpp"
#include <ap_int.h>

using uint3_t = ap_uint<3>;
using uint6_t = ap_uint<6>;
using uint36_t = ap_uint<36>;

const ap_uint<1> b0w1 = 0;
const ap_uint<2> b0w2 = 0;

// @thanks https://acri-vhls-challenge.web.app/user/@Ryuz88/code/bF8U1WQa7LxBNKl0HXzf
// @see https://fpga.org/2014/09/05/quick-fpga-hacks-population-count/
//
// 6:3 compressor as a 64x3b ROM -- three 6-LUTs
//
void c63(uint6_t i, uint3_t& o) {
	static const uint3_t table[] = {
		0, 1, 1, 2,
		1, 2, 2, 3,
		1, 2, 2, 3,
		2, 3, 3, 4,
		1, 2, 2, 3,
		2, 3, 3, 4,
		2, 3, 3, 4,
		3, 4, 4, 5,
		1, 2, 2, 3,
		2, 3, 3, 4,
		2, 3, 3, 4,
		3, 4, 4, 5,
		2, 3, 3, 4,
		3, 4, 4, 5,
		3, 4, 4, 5,
		4, 5, 5, 6,
	};
	o = table[i];
}

// Resource usage
//   FF   : 849
//   LUT  : 4586
//   DSP  : 0
//   BRAM : 2
//   URAM : 0
// Clock period (ns): 2.431
// Clock frequency (MHz): 411
// Simulation cycle: 515
// Simulation time (ns): 1251.965
int16_t pop36(uint36_t i) {
	uint3_t c0500, c1106, c1712, c2318, c2924, c3530, c0, c1, c2;

	c63(i( 5,  0), c0500);
	c63(i(11,  6), c1106);
	c63(i(17, 12), c1712);
	c63(i(23, 18), c2318);
	c63(i(29, 24), c2924);
	c63(i(35, 30), c3530);

	c63((c0500[0], c1106[0], c1712[0], c2318[0], c2924[0], c3530[0]), c0);
	c63((c0500[1], c1106[1], c1712[1], c2318[1], c2924[1], c3530[1]), c1);
	c63((c0500[2], c1106[2], c1712[2], c2318[2], c2924[2], c3530[2]), c2);

	uint16_t sum = (b0w1, c0) + (b0w1, c1, b0w1) + (c2, b0w2);
	return sum;
}

// Resource usage
//   FF   : 779
//   LUT  : 1502
//   DSP  : 0
//   BRAM : 2
//   URAM : 0
// Clock period (ns): 2.431
// Clock frequency (MHz): 411
// Simulation cycle: 643
// Simulation time (ns): 1563.133
int16_t reference_implement(uint32_t x) {
	int16_t sum = 0;
	ap_uint<32> v = x;
	for (int i = 0; i < 32; i++) {
		if (v[i] == 1) sum++;
	}
	return sum;
}

// @see HD, Figure 5-1 Counting 1-bits
//
// Resource usage
//   FF   : 881
//   LUT  : 1656
//   DSP  : 0
//   BRAM : 2
//   URAM : 0
// Clock period (ns): 2.431
// Clock frequency (MHz): 411
// Simulation cycle: 515
// Simulation time (ns): 1251.965
int16_t hackers_delight(uint32_t x) {
	x = (x & 0x55555555) + ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	x = (x & 0x0f0f0f0f) + ((x >> 4) & 0x0f0f0f0f);
	x = (x & 0x00ff00ff) + ((x >> 8) & 0x00ff00ff);
	x = (x & 0x0000ffff) + ((x >> 16) & 0x0000ffff);
	return x;
}

void kernel(uint32_t in, int16_t* out) {
	//*out = reference_implement(in);
	//*out = hackers_delight(in);
	*out = pop36(in);
}
