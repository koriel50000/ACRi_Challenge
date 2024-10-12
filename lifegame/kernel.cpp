#include <stdio.h>
#include "kernel.hpp"

void next_step(const ap_uint<1> c[64], ap_uint<1> n[64]) {
	y_loop: for (int y = 0; y < 8; y++) {
#pragma HLS unroll
		x_loop: for (int x = 0; x < 8; x++) {
			int nb =  c[(y-1 & 7) * 8 + (x-1 & 7)]
				+ c[(y-1 & 7) * 8 + (x   & 7)]
				+ c[(y-1 & 7) * 8 + (x+1 & 7)]
				+ c[(y   & 7) * 8 + (x-1 & 7)]
				+ c[(y   & 7) * 8 + (x+1 & 7)]
				+ c[(y+1 & 7) * 8 + (x-1 & 7)]
				+ c[(y+1 & 7) * 8 + (x   & 7)]
				+ c[(y+1 & 7) * 8 + (x+1 & 7)];
		        int tt = nb ^ 0x03;
			if (tt == 0 || (tt ^ c[y * 8 + x]) == 0) {
				n[y * 8 + x] = 1;
			} else {
				n[y * 8 + x] = 0;
			}
			//printf(c[y * 8 + x] == 1 ? " O" : " .");
		}
		//printf("\n");
	}
	//printf("\n");
}

void kernel(const int generation, const ap_uint<1> in[64], ap_uint<1> out[64]) {
#pragma HLS iterface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in
#pragma HLS array_partition variable=out

	ap_uint<1> odd[64], even[64];
#pragma HLS array_partition variable=odd
#pragma HLS array_partition variable=even

	load_loop: for (int i = 0; i < 64; i++) {
#pragma HLS unroll
		odd[i] = in[i];
	}

	generation_loop: for (int j = 0; j < generation; j += 2) {
#pragma HLS pipeline II=1
		next_step(odd, even);
		next_step(even, odd);
	}

	store_loop: for (int i = 0; i < 64; i++) {
#pragma HLS unroll
		out[i] = odd[i];
	}
}
