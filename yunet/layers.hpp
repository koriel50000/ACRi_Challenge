#pragma once
#include <hls_vector.h>
#undef INLINE
#include <hls_stream.h>
#include "types.hpp"
#include "bitarith.hpp"
#include "window_buffer.hpp"

const int WIDTH = 80;
const int HEIGHT = 80;
const int CHANNEL = 64;
const int FILTER = 64;

const int KERNEL = 3;
const int THRESHOLD = 14;

const int FEAT_WIDTH = WIDTH / 4;
const int FEAT_HEIGHT = HEIGHT / 4;

using uint4_t = ap_uint<4>;
using data_t = int_t<CHANNEL>;
using block_data_t = data_t[HEIGHT * WIDTH * 1];
using block_feat_t = data_t[FEAT_HEIGHT * FEAT_WIDTH * 1];
using block_conv_t = data_t[FILTER * 1 * KERNEL * KERNEL];
using block_thr_t = int16_t[FILTER][THRESHOLD];
using win_t = hls::vector<data_t, KERNEL * KERNEL>;
using linebuf_t = LineBuffer<256, KERNEL, data_t, win_t>;

template <typename T>
using fifo = hls::stream<T>;

template <int H, int W, int C, int F, int KN, int ST>
class Conv2Dstride {
public:
	void windowize(linebuf_t& linebuf, fifo<uint64_t>& ins, fifo<win_t>& pips) {
		linebuf.reset(W + KN - 1);

		int x = 0 - (KN - 1) / 2;
		int y = 0 - (KN - 1) / 2;
		windowize: for (int i = 0; i < (W + KN - 1) * (H + KN - 1); i++) {
#pragma HLS pipeline
			// input
			data_t val;
			if (0 <= x && x < W	&& 0 <= y && y < H) {
				val = ins.read();
			} else {
				val = 0;
			}
			// buffering
			if (i < (W + KN - 1) * (KN - 1)) {
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
			if (x >= W + (KN - 1) / 2) {
				x = 0 - (KN - 1) / 2;
				y++;
			}
		}
	}

	void compute(const int h, const int w, const int f,
		block_conv_t& wi, block_thr_t& thr, fifo<win_t>& pips, block_data_t& outb)
	{
		compute_h: for (int y = 0; y < H; y++) {
			if (y < h) {
				compute_w: for (int x = 0; x < W; x++) {
					if (x < w) {
						win_t ival = pips.read();
						data_t oval = 0;
						compute_f: for (int j = 0; j < F; j++) {
#pragma HLS pipeline
							if (j < f) {
								int16_t acc = 0;
								compute_k: for (int k = 0; k < KN * KN; k++) {
									acc += muladd<C>(ival[k], wi[j * KN * KN + k]);
								}
								oval[j] = batch_norm_relu(acc, thr[j]);
							}
						}
						outb[y * WIDTH + x] = oval;
					}
				}
			}
		}
	}
};

template <int H, int W, int C, int F, int KN>
class Conv2Ddepthwise {
public:
	void windowize(const int h, const int w, linebuf_t& linebuf, block_data_t& inb, fifo<win_t>& pips) {
		linebuf.reset(w + KN - 1);

		int x = 0 - (KN - 1) / 2;
		int y = 0 - (KN - 1) / 2;
		windowize: for (int i = 0; i < (W + KN - 1) * (H + KN - 1); i++) {
#pragma HLS pipeline
			// @see UG1399, Vitis HLS Coding Styles > Loops > Variable Loop Bounds
			if (i < (w + KN - 1) * (h + KN - 1)) {
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
				if ((KN - 1) / 2 <= x && (KN - 1) / 2 <= y) {
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
	}

	void compute(const int h, const int w, const int f, const bool relu,
		block_conv_t& wi, block_thr_t& thr, fifo<win_t>& pips, block_data_t& outb)
	{
		compute_h: for (int y = 0; y < H; y++) {
			if (y < h) {
				compute_w: for (int x = 0; x < W; x++) {
					if (x < w) {
						win_t ival = pips.read();
						data_t oval = 0;
						compute_f: for (int j = 0; j < F; j++) {
#pragma HLS pipeline
							if (j < f) {
								data_t val = 0;
								compute_k: for (int k = 0; k < KN * KN; k++) {
#pragma HLS unroll
									val[k] = ival[k][j];
								}
								int16_t acc = muladd<C>(val, wi[j]);
								if (relu) {
									oval[j] = batch_norm_relu(acc, thr[j]);
								} else {
									oval[j] = batch_norm(acc, thr[j]);
								}
							}
						}
						outb[y * WIDTH + x] = oval;
					}
				}
			}
		}
	}

	void compute_branch(const int h, const int w, const int f,
		block_conv_t& wi, block_thr_t& thr, fifo<win_t>& pips,
		block_data_t& outb, block_feat_t& outfb)
	{
		compute_h: for (int y = 0; y < H; y++) {
			if (y < h) {
				compute_w: for (int x = 0; x < W; x++) {
					if (x < w) {
						win_t ival = pips.read();
						data_t oval = 0;
						compute_f: for (int j = 0; j < F; j++) {
#pragma HLS pipeline
							if (j < f) {
								data_t val = 0;
								compute_k: for (int k = 0; k < KN * KN; k++) {
#pragma HLS unroll
									val[k] = ival[k][j];
								}
								int16_t acc = muladd<C>(val, wi[j]);
								oval[j] = batch_norm_relu(acc, thr[j]);
							}
						}
						outb[y * WIDTH + x] = oval;
						outfb[y * FEAT_WIDTH + x] = oval;
					}
				}
			}
		}
	}
};

template <int H, int W, int C, int F>
class Conv2Dpointwise {
public:
	void compute(const int h, const int w, const int f,
		block_conv_t& wi, block_thr_t& thr, block_data_t& buf)
	{
		compute_h: for (int y = 0; y < H; y++) {
			if (y < h) {
				compute_w: for (int x = 0; x < W; x++) {
					if (x < w) {
						data_t ival = buf[y * WIDTH + x];
						data_t oval = 0;
						compute_f: for (int j = 0; j < F; j++) {
#pragma HLS pipeline
							if (j < f) {
								int16_t acc = muladd<C>(ival, wi[j]);
								oval[j] = batch_norm(acc, thr[j]);
							}
						}
						buf[y * WIDTH + x] = oval;
					}
				}
			}
		}
	}

	void compute(const int h, const int w, const int f,
		block_conv_t& wi, block_thr_t& thr, block_feat_t& infb, block_data_t& outb)
	{
		compute_h: for (int y = 0; y < H; y++) {
			if (y < h) {
				compute_w: for (int x = 0; x < W; x++) {
					if (x < w) {
						data_t ival = infb[y * FEAT_WIDTH + x];
						data_t oval = 0;
						compute_f: for (int j = 0; j < F; j++) {
#pragma HLS pipeline
							if (j < f) {
								int16_t acc = muladd<C>(ival, wi[j]);
								oval[j] = batch_norm(acc, thr[j]);
							}
						}
						outb[y * WIDTH + x] = oval;
					}
				}
			}
		}
	}

	void fuse_topdown_compute(const int h, const int w, const int f,
		block_conv_t& wi, block_thr_t& thr, block_feat_t& infb, block_data_t& buf)
	{
		block_feat_t tmpb;
		compute_h: for (int y = 0; y < H; y++) {
			if (y < h) {
				compute_w: for (int x = 0; x < W; x++) {
					if (x < w) {
						data_t ivald = infb[y * FEAT_WIDTH + x];
						tmpb[y * FEAT_WIDTH + x] = buf[y * WIDTH + x];
						data_t ivalt = tmpb[y / 2 * FEAT_WIDTH + x / 2]; // resize by 2x
						data_t oval = 0;
						compute_f: for (int j = 0; j < F; j++) {
#pragma HLS pipeline
							if (j < f) {
								int16_t acc = 0;
								acc += muladd<C>(ivald, wi[j]);
								acc += muladd<C>(ivalt, wi[j]);
								oval[j] = batch_norm(acc, thr[j]);
							}
						}
						buf[y * WIDTH + x] = oval;
					}
				}
			}
		}
	}
};

template <int H, int W, int C>
class MaxPool2x2 {
private:
	void maxpool(const data_t v1, const data_t v2, data_t& ov) {
		maxpool: for (int z = 0; z < C; z++) {
#pragma HLS unroll
			ov[z] = (v1[z] > v2[z]) ? v1[z] : v2[z];
		}
	}
public:
	void compute_h(const int h, const int w,
		block_data_t& inb, fifo<data_t>& pips)
	{
		compute_h: for (int y = 0; y < H; y++) {
			if (y >= h) break;
			compute_w: for (int x = 0; x < W; x += 2) {
#pragma HLS pipeline
				if (x >= w) break;
				data_t val1 = inb[y * WIDTH + x];
				data_t val2 = inb[y * WIDTH + x + 1];
				data_t oval;
				maxpool(val1, val2, oval);
				pips.write(oval);
			}
		}
	}

	void compute_v(const int oh, const int ow,
		fifo<data_t>& pips, block_data_t& outb)
	{
		static data_t buf[W / 2];
#pragma HLS array_partition variable=buf

		compute_h: for (int y = 0; y < H; y++) {
			if (y >= oh) break;
			compute_w1: for (int x = 0; x < W; x++) {
#pragma HLS pipeline
				if (x >= ow) break;
				buf[x] = pips.read();
			}
			compute_w2: for (int x = 0; x < W; x++) {
#pragma HLS pipeline
				if (x >= ow) break;
				data_t val1 = buf[x];
				data_t val2 = pips.read();
				data_t oval;
				maxpool(val1, val2, oval);
				outb[y * WIDTH + x] = oval;
			}
		}
	}
};
