#pragma once
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

using data_t = int_t<CHANNEL>;
using block_data_t = data_t[HEIGHT * WIDTH * 1];
using block_feat_t = data_t[FEAT_HEIGHT * FEAT_WIDTH * 1];
using block_conv_t = data_t[FILTER * 1 * KERNEL * KERNEL];
using block_thr_t = int16_t[FILTER][THRESHOLD];
using data_ptr_t = data_t*;

enum ConvOp {
	SpatialOp,
	DepthwiseOp,
	DepthwiseBrOp,
	DepthwiseHeadOp,
	SingleChannelOp,
	PointwiseOp,
	PointwiseHeadOp,
	FuseTopDownOp,
};

struct ConvMode {
	ConvOp op;
	int stride;
	int input;
	int muladd;
	bool depthwise;
	bool relu;
	int alpha;
	int beta;
};

static const ConvMode Spatial = { SpatialOp, 2, 9, 9, false, true, 9, 1 };
static const ConvMode Depthwise = { DepthwiseOp, 1, 10, 1, true, true, 1, 0 };
static const ConvMode DepthwiseBr = { DepthwiseBrOp, 1, 10, 1, true, true, 1, 0 };
static const ConvMode DepthwiseHead = { DepthwiseHeadOp, 1, 10, 1, true, true, 1, 0 };
static const ConvMode SingleChannel = { SingleChannelOp, 1, 10, 1, true, false, 1, 0 };
static const ConvMode Pointwise = { PointwiseOp, 0, 1, 1, false, false, 1, 0 };
static const ConvMode PointwiseHead = { PointwiseHeadOp, 0, 1, 1, false, false, 1, 0 };
static const ConvMode FuseTopDown = { FuseTopDownOp, 0, 2, 2, false, false, 1, 0 };

namespace Conv2D {
	void windowize(const int h, const int w, const int st, const bool depthwise,
		block_data_t& inb, fifo<data_t>& outs);

	void compute(const int h, const int w, const int f, const ConvMode& type,
		const block_conv_t& wi, const block_thr_t& thr, fifo<data_t>& ins, fifo<data_t>& outs);
}

namespace MaxPool2x2 {
	void compute_h(const int h, const int w, data_ptr_t inb, data_ptr_t outb);

	void compute_v(const int h, const int w, data_ptr_t inb, data_ptr_t outb);
}
