#pragma once
#include <hls_vector.h>
#undef INLINE
#include <hls_stream.h>
#include "types.hpp"
#include "arith.hpp"
#include "window_buffer.hpp"

const int WIDTH = 28;
const int HEIGHT = 28;
const int CHANNEL = 16;
const int FILTER = 16;

const int KERNEL = 3;
const int THRESHOLD = 14;

const int FLATTEN = 784;
const int CLASS = 10;
const int CHUNK_SIZE = 16;  // == FILTER

using data_t = int_t<CHANNEL>;
using block_data_t = data_t[HEIGHT * WIDTH * 1];
using block_conv_t = data_t[FILTER * 1 * KERNEL * KERNEL];
using block_thr_t = int16_t[FILTER][THRESHOLD];
using block_mat_t = data_t[CLASS * FLATTEN / CHUNK_SIZE];
using win_t = hls::vector<data_t, KERNEL * KERNEL>;
using linebuf_t = LineBuffer32<KERNEL, data_t, win_t>;

template <typename T>
using fifo = hls::stream<T>;

template <int H, int W, int C, int F, int KN, bool RELU, int ST = 1>
class Conv2D {
public:
	void windowize(const int h, const int w, linebuf_t& linebuf, block_data_t& inb, fifo<win_t>& pips) {
		linebuf.init(w + KN - 1);

        int x = 0 - (KN - 1) / 2;
        int y = 0 - (KN - 1) / 2;
		for (int i = 0; i < (W + KN - 1) * (H + KN - 1); i++) {
#pragma HLS pipeline
			// @see UG1399, Vitis HLS Coding Styles > Loops > Variable Loop Bounds
			if (i >= (w + KN - 1) * (h + KN - 1)) break;
   			// input
   			data_t val;
    		if (0 <= x && x < w	&& 0 <= y && y < h) {
	    		val = inb[y * WIDTH + x];
		    } else {
			    val = 0;
   			}
           // buffering
   			if (i < (w + KN - 1) * (KN - 1)) {
    			linebuf.insert_linebuf(val);
	    	} else {
			    linebuf.slide_window(val);
   			}
 			// output
   			if ((KN - 1) / 2 <= x && (KN - 1) / 2 <= y
   			    && (x - (KN - 1) / 2) % ST == 0 && (y - (KN - 1) / 2) % ST == 0)
   			{
    			win_t oval = linebuf.get_window();
	    		pips.write(oval);
	    	}
		    x++;
		    if (x >= w + (KN - 1) / 2) {
                x = 0 - (KN - 1) / 2;
		        y++;
		    }
		}
	}

	void compute(const int h, const int w, const int c, const int f,
		block_conv_t& wi, block_thr_t& thr,
		fifo<win_t>& pips, block_data_t& outb)
	{
		for (int y = 0; y < H; y++) {
			if (y >= h) break;
			for (int x = 0; x < W; x++) {
				if (x >= w) break;
				win_t val = pips.read();
				data_t oval;
				for (int j = 0; j < F; j++) {
#pragma HLS pipeline
					if (j >= f) break;
					int16_t acc = 0;
					for (int k = 0; k < KN * KN; k++) {
						acc += muladd<C>(c, val[k], wi[j * KN * KN + k]);
					}
					if (RELU) {
    					oval[j] = batch_norm_relu(acc, thr[j]);
	   				} else {
		    			oval[j] = batch_norm(acc, thr[j]);
					}
				}
				outb[y * WIDTH + x] = oval;
			}
		}
	}
};

template <int H, int W, int C, int F>
class Conv2D1x1 {
public:
	void compute(const int h, const int w, const int c, const int f,
		block_conv_t& wi, block_thr_t& thr,
		block_data_t& inb, block_data_t& outb)
	{
		for (int y = 0; y < H; y++) {
			if (y >= h) break;
			for (int x = 0; x < W; x++) {
				if (x >= w) break;
				data_t val = inb[y * WIDTH + x];
				data_t oval;
				for (int j = 0; j < F; j++) {
#pragma HLS pipeline
					if (j >= f) break;
					int16_t acc = muladd<C>(c, val, wi[j]);
					oval[j] = batch_norm(acc, thr[j]);
				}
				outb[y * WIDTH + x] = oval;
			}
		}
	}
};

template <int H, int W, int C>
class MaxPool2x2 {
private:
	void maxpool(const int c, const data_t v1, const data_t v2, data_t& ov) {
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			if (z >= c) break;
			ov[z] = (v1[z] > v2[z]) ? v1[z] : v2[z];
		}
	}
public:
	void compute_h(const int h, const int w, const int c,
		block_data_t& inb, fifo<data_t>& pips)
	{
		for (int y = 0; y < H; y++) {
			if (y >= h) break;
			for (int x = 0; x < W; x += 2) {
#pragma HLS pipeline
				if (x >= w) break;
				data_t val1 = inb[y * WIDTH + x];
				data_t val2 = inb[y * WIDTH + x + 1];
				data_t oval;
				maxpool(c, val1, val2, oval);
				pips.write(oval);
			}
		}
	}

	void compute_v(const int oh, const int ow, const int oc,
		fifo<data_t>& pips, block_data_t& outb)
	{
		static data_t buf[W / 2];
#pragma HLS array_partition variable=buf

		for (int y = 0; y < H; y++) {
			if (y >= oh) break;
			for (int x = 0; x < W; x++) {
#pragma HLS pipeline
				if (x >= ow) break;
				buf[x] = pips.read();
			}
			for (int x = 0; x < W; x++) {
#pragma HLS pipeline
				if (x >= ow) break;
				data_t val1 = buf[x];
				data_t val2 = pips.read();
				data_t oval;
				maxpool(oc, val1, val2, oval);
				outb[y * WIDTH + x] = oval;
			}
		}
	}
};

template <int CL, int FL, int K, int H, int W>
class Dense {
private:
	using IT = int_t<K>;
	using OT = int_t<CL,16>;  // int16_t
public:
	void flatten(block_mat_t& mat, block_data_t& inb, fifo<OT>& pips) {
		int ptr = 0;
		for (int y = 0; y < H; y++) {
			for (int x = 0; x < W; x++) {
				IT vu = inb[y * WIDTH + x];
				OT oval;
				for (int i = 0; i < CL; i++) {
#pragma HLS pipeline
					IT wi = mat[ptr++];
					int16_t acc = muladd<K>(K, vu, wi);
					oval[i] = acc;
				}
				pips.write(oval);
			}
		}
	}

	void write_result(int out[1], fifo<OT>& pips) {
		static int16_t acc[CL];
#pragma HLS array_partition variable=acc

		for (int i = 0; i < CL; i++) {
#pragma HLS unroll
			acc[i] = 0;
		}

		for (int j = 0; j < FL / K; j++) {
#pragma HLS pipeline
			OT val = pips.read();
			for (int i = 0; i < CL; i++) {
#pragma HLS unroll
				acc[i] += val[i];
			}
		}

		int16_t max = INT16_MIN;
		int m = 0;
		for (int i = 0; i < CL; i++) {
#pragma HLS pipeline
			if (acc[i] > max) {
				max = acc[i];
				m = i;
			}
		}
		out[0] = m;
	}
};
