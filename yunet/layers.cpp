#include "layers.hpp"

namespace Conv2D {
	void windowize(const int h, const int w, const int st, const bool depthwise,
		block_data_t& inb, fifo<data_t>& outs)
	{
		Window<KERNEL, KERNEL, data_t> winb;

		windowize_h: for (int y = 0; y < HEIGHT * 2; y++) {
			// @see UG1399, Vitis HLS Coding Styles > Loops > Variable Loop Bounds
			if (y < h) {
				winb.reset_center_col();
				winb.insert_right_col(h, w, depthwise, inb, y, 0);
				windowize_w: for (int x = 0; x < WIDTH * 2; x++) {
					if (x < w) {
						// buffering
						winb.shift_pixels_left();
						winb.insert_right_col(h, w, depthwise, inb, y, x + 1);
			 			// output
						if (x % st == 0 && y % st == 0) {
							if (depthwise) {
								outs.write(data_t(0));
							}
							for (int k = 0; k < KERNEL * KERNEL; k++) {
								outs.write(winb[k]);
							}
						}
					}
				}
			}
		}
	}

	void compute(const int h, const int w, const int f, const ConvMode& mode,
		const block_conv_t& wi, const block_thr_t& thr, fifo<data_t>& ins, fifo<data_t>& outs)
	{
		compute_hw: for (int i = 0; i < HEIGHT * WIDTH; i++) {
			if (i < h * w) {
				// input
				data_t inputs[KERNEL * KERNEL + 1];
#pragma HLS array_partition variable=inputs complete

				for (int k = 0; k < KERNEL * KERNEL + 1; k++) {
					if (k < mode.input) {
						inputs[k] = ins.read();
					}
				}

				data_t oval = 0;
				compute_f: for (int j = 0; j < FILTER; j++) {
					if (j < f) {
						if (mode.depthwise) {
							compute_k2: for (int k = 1; k < KERNEL * KERNEL + 1; k++) {
#pragma HLS unroll
								inputs[0][k - 1] = inputs[k][j];
							}
						}

						// convolution
						int16_t acc = 0;
						compute_k: for (int k = 0; k < KERNEL * KERNEL; k++) {
#pragma HLS pipeline off
							if (k < mode.muladd) {
								acc += muladd<CHANNEL>(inputs[k], wi[j * mode.alpha + k * mode.beta]);
							}
						}

						// batch norm
						if (mode.relu) {
							oval[j] = batch_norm_relu(acc, thr[j]);
						} else {
							oval[j] = batch_norm(acc, thr[j]);
						}
					}
				}
				outs.write(oval);
			}
		}
	}
}

namespace MaxPool2x2 {
	void maxpool(const data_t v1, const data_t v2, data_t& ov) {
#pragma HLS inline off
		maxpool: for (int z = 0; z < CHANNEL; z++) {
#pragma HLS unroll
			ov[z] = (v1[z] > v2[z]) ? v1[z] : v2[z];
		}
	}

	void compute_h(const int h, const int w, data_ptr_t inb, data_ptr_t outb) {
		compute_hw: for (int i = 0; i < HEIGHT * WIDTH / 2; i++) {
			if (i < h * w / 2) {
				data_t val1 = *inb++;
				data_t val2 = *inb++;
				data_t oval;
				maxpool(val1, val2, oval);
				*outb++ = oval;
			}
		}
	}

	void compute_v(const int h, const int w, data_ptr_t inb, data_ptr_t outb) {
		compute_h: for (int y = 0; y < HEIGHT; y++) {
			if (y < h) {
				data_ptr_t inb1 = inb;
				data_ptr_t inb2 = inb + w;
				compute_w: for (int x = 0; x < WIDTH; x++) {
					if (x < w) {
						data_t val1 = *inb1++;
						data_t val2 = *inb2++;
						data_t oval;
						maxpool(val1, val2, oval);
						*outb++ = oval;
					}
				}
				inb = inb2;
			}
		}
	}
}
