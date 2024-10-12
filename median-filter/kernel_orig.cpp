#include "kernel.hpp"
#include <hls_stream.h>

typedef hls::stream<uint8_t> fifo_t;

void insert_sort2(uint8_t rank[2], const uint8_t v) {
	bool b0 = (v > rank[0]);
	bool b1 = (v > rank[1]);
	if (b0) {
		rank[1] = rank[0];
		rank[0] = v;
	} else if (b1) {
		rank[1] = v;
	}
}

void insert_sort3(uint8_t rank[3], const uint8_t v) {
	bool b0 = (v > rank[0]);
	bool b1 = (v > rank[1]);
	bool b2 = (v > rank[2]);
	if (b0) {
		rank[2] = rank[1];
		rank[1] = rank[0];
		rank[0] = v;
	} else if (b1) {
		rank[2] = rank[1];
		rank[1] = v;
	} else if (b2) {
		rank[2] = v;
	}
}

void insert_sort5(uint8_t rank[5], const uint8_t v) {
	bool b0 = (v > rank[0]);
	bool b1 = (v > rank[1]);
	bool b2 = (v > rank[2]);
	bool b3 = (v > rank[3]);
	bool b4 = (v > rank[4]);
	if (b0) {
		rank[4] = rank[3];
		rank[3] = rank[2];
		rank[2] = rank[1];
		rank[1] = rank[0];
		rank[0] = v;
	} else if (b1) {
		rank[4] = rank[3];
		rank[3] = rank[2];
		rank[2] = rank[1];
		rank[1] = v;
	} else if (b2) {
		rank[4] = rank[3];
		rank[3] = rank[2];
		rank[2] = v;
	} else if (b3) {
		rank[4] = rank[3];
		rank[3] = v;
	} else if (b4) {
		rank[4] = v;
	}
}

uint8_t corner_median(const uint8_t v[4]) {
	uint8_t rank[2] = { 0, 0 };
	neighbor_loop4: for (int i = 0; i < 4; i++) {
		insert_sort2(rank, v[i]);
	}
	return rank[1];
}

uint8_t edge_median(const uint8_t v[6]) {
	uint8_t rank[3] =  { 0, 0, 0 };
	neighbor_loop6: for (int i = 0; i < 6; i++) {
		insert_sort3(rank, v[i]);
	}
	return rank[2];
}

uint8_t inside_median(const uint8_t v[9]) {
	uint8_t rank[5] = { 0, 0, 0, 0, 0 };
	neighbor_loop9: for (int i = 0; i < 9; i++) {
		insert_sort5(rank, v[i]);
	}
	return rank[4];
}

// TODO hls::LineBuffer, hls::Window
// @see https://marsee101.blog.fc2.com/blog-entry-3418.html
// @see https://github.com/hashi0203/Vitis_HLS_Gaussian/blob/main/src/gaussian.cpp
void top(const uint8_t* mline, uint8_t* lline, fifo_t& ins, fifo_t& outs) {
	ins >> lline[0];
	ins >> lline[1];
	uint8_t tl[] = {
		mline[0], mline[1],
		lline[0], lline[1]
	};
	outs << corner_median(tl);
	for (int x = 1; x < WIDTH - 1; x++) {
		ins >> lline[x + 1];
		uint8_t tc[] = {
			mline[x - 1], mline[x], mline[x + 1],
			lline[x - 1], lline[x], lline[x + 1]
		};
		outs << edge_median(tc);
	}
	uint8_t tr[] = {
		mline[WIDTH - 2], mline[WIDTH - 1],
		lline[WIDTH - 2], lline[WIDTH - 1]
	};
	outs << corner_median(tr);
}

void middle(const uint8_t* uline, const uint8_t* mline, uint8_t* lline, fifo_t& ins, fifo_t& outs) {
	ins >> lline[0];
	ins >> lline[1];
	uint8_t ml[] = {
		uline[0], uline[1],
		mline[0], mline[1],
		lline[0], lline[1]
	};
	outs << edge_median(ml);
	for (int x = 1; x < WIDTH - 1; x++) {
		ins >> lline[x + 1];
		uint8_t mc[] = {
			uline[x - 1], uline[x], uline[x + 1],
			mline[x - 1], mline[x], mline[x + 1],
			lline[x - 1], lline[x], lline[x + 1]
		};
		outs << inside_median(mc);
	}
	uint8_t mr[] = {
		uline[WIDTH - 2], uline[WIDTH - 1],
		mline[WIDTH - 2], mline[WIDTH - 1],
		lline[WIDTH - 2], lline[WIDTH - 1]
	};
	outs << edge_median(mr);
}

void bottom(const uint8_t* uline, const uint8_t* mline, fifo_t& outs) {
	uint8_t bl[] = {
		uline[0], uline[1],
		mline[0], mline[1]
	};
	outs << corner_median(bl);
	for (int x = 1; x < WIDTH - 1; x++) {
		uint8_t bc[] = {
			uline[x - 1], uline[x], uline[x + 1],
			mline[x - 1], mline[x], mline[x + 1]
		};
		outs << edge_median(bc);
	}
	uint8_t br[] = {
		uline[WIDTH - 2], uline[WIDTH - 1],
		mline[WIDTH - 2], mline[WIDTH - 1]
	};
	outs << corner_median(br);
}

void load(fifo_t& ins, uint8_t* line) {
	for (int x = 0; x < WIDTH; x++) {
		ins >> line[x];
	}
}

void shift_down(uint8_t* uline, uint8_t* mline, const uint8_t* lline) {
	for (int x = 0; x < WIDTH; x++) {
#pragma HLS unroll
		uline[x] = mline[x];
		mline[x] = lline[x];
	}
}

void read_input(const uint8_t in[WIDTH * HEIGHT], fifo_t& ins) {
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
#pragma HLS unroll factor=128
		ins << in[i];
	}
}

void compute_median(fifo_t& ins, fifo_t& outs) {
	// @see Vivado HLS 最適化手法ガイド
	// https://www.xilinx.com/support/documentation/sw_manuals_j/xilinx2017_4/ug1270-vivado-hls-opt-methodology-guide.pdf
	// データ アクセス パターン
	uint8_t uline[WIDTH], mline[WIDTH], lline[WIDTH];
#pragma HLS array_pertition variable=uline
#pragma HLS array_pertition variable=mline
#pragma HLS array_pertition variable=lline

	load(ins, uline);
	// line 0, line 1
	top(uline, mline, ins, outs);

	for (int y = 1; y < HEIGHT - 1; y++) {
		// line y - 1, line y, line y + 1
		middle(uline, mline, lline, ins, outs);
		shift_down(uline, mline, lline);
	}

	// line HEIGHT - 2, line HEIGHT - 1
	bottom(uline, mline, outs);
}

void write_result(uint8_t out[WIDTH * HEIGHT], fifo_t& outs) {
	for (int i = 0; i < WIDTH * HEIGHT; i++) {
		outs >> out[i];
	}
}

void kernel(const uint8_t in[WIDTH * HEIGHT], uint8_t out[WIDTH * HEIGHT]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_pertition variable=in cyclic factor=128

	fifo_t ins("input_fifo");
	fifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input(in, ins);
	compute_median(ins, outs);
	write_result(out, outs);
}
