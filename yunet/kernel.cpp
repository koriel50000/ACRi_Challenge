#include "kernel.hpp"
#include "layers.hpp"
#include "predictions.hpp"

using pred_t = Predictions<20, 20, 10, data_t>;

void read_input(fifo<axis_data64>& ins, block_data_t& outb) {
#pragma HLS inline off
	read_input_hw: for (int i = 0; i < HEIGHT * WIDTH; i++) {
		uint64_t w0 = ins.read().data;
		uint64_t w1 = ins.read().data;
		uint64_t w2 = ins.read().data;
		uint64_t w3 = ins.read().data;
		outb[i] = data_t(w3, w2, w1, w0);
	}
}

void read_weight(const int f, const int kn, const bool concat,
	fifo<axis_data64>& ins, block_conv_t& outw, block_thr_t& outh)
{
#pragma HLS inline off
	read_weight_f: for (int i = 0; i < FILTER * KERNEL * KERNEL; i++) {
		if (i < f * kn * kn) {
			if (concat) {
				uint64_t w3 = ins.read().data;
				uint64_t w2 = ins.read().data;
				uint64_t w1 = ins.read().data;
				uint64_t w0 = ins.read().data;
				outw[i] = data_t(w3, w2, w1, w0);
			} else {
				uint64_t w = ins.read().data;
				outw[i] = data_t(w);
			}
		}
	}

	read_threshold_f: for (int j = 0; j < FILTER; j++) {
		if (j < f) {
			uint64_t w0 = ins.read().data;
			uint64_t w1 = ins.read().data;
			uint64_t w2 = ins.read().data;
			uint64_t w3 = ins.read().data;

			outh[j][0]  = (w0 >> 48) & 0xffff;
			outh[j][1]  = (w0 >> 32) & 0xffff;
			outh[j][2]  = (w0 >> 16) & 0xffff;
			outh[j][3]  = (w0      ) & 0xffff;
			outh[j][4]  = (w1 >> 48) & 0xffff;
			outh[j][5]  = (w1 >> 32) & 0xffff;
			outh[j][6]  = (w1 >> 16) & 0xffff;
			outh[j][7]  = (w1      ) & 0xffff;
			outh[j][8]  = (w2 >> 48) & 0xffff;
			outh[j][9]  = (w2 >> 32) & 0xffff;
			outh[j][10] = (w2 >> 16) & 0xffff;
			outh[j][11] = (w2      ) & 0xffff;
			outh[j][12] = (w3 >> 48) & 0xffff;
			outh[j][13] = (w3 >> 32) & 0xffff;
		}
	}
}

void write_output(pred_t& preds, fifo<axis_data8>& outs) {
#pragma HLS inline off
	uint8_t size = preds.get_bboxes();
	axis_data8 pkt;
	pkt.data = size;
	pkt.last = 1;
	outs.write(pkt);
	write_output: for (int i = 0; i < MAX_DETECTIONS; i++) {
		if (i < size) {
			const Detection& detect = preds.get_detection(i);
			pkt.data = detect.x1;
			pkt.last = 0;
			outs.write(pkt);
			pkt.data = detect.y1;
			outs.write(pkt);
			pkt.data = detect.x2;
			outs.write(pkt);
			pkt.data = detect.y2;
			outs.write(pkt);
			pkt.data = (detect.score >> 8) & 0xff;
			outs.write(pkt);
			pkt.data = detect.score & 0xff;
			outs.write(pkt);
			pkt.data = detect.kps[0];
			outs.write(pkt);
			pkt.data = detect.kps[1];
			outs.write(pkt);
			pkt.data = detect.kps[2];
			outs.write(pkt);
			pkt.data = detect.kps[3];
			outs.write(pkt);
			pkt.data = detect.kps[4];
			outs.write(pkt);
			pkt.data = detect.kps[5];
			outs.write(pkt);
			pkt.data = detect.kps[6];
			outs.write(pkt);
			pkt.data = detect.kps[7];
			outs.write(pkt);
			pkt.data = detect.kps[8];
			outs.write(pkt);
			pkt.data = detect.kps[9];
			pkt.last = 1;
			outs.write(pkt);
		}
	}
}

void branch_to_feature(const int h, const int w, fifo<data_t>& ins, block_data_t& outb, block_feat_t& outfb) {
	copy_to_feature_hw: for (int i = 0; i < FEAT_HEIGHT * FEAT_WIDTH; i++) {
		if (i < h * w) {
			data_t val = ins.read();
			outb[i] = val;
			outfb[i] = val;
		}
	}
}

void array_to_stream(const int h, const int w, data_ptr_t inb, fifo<data_t>& outs) {
	array_to_stream_hw: for (int i = 0; i < HEIGHT * WIDTH; i++) {
		if (i < h * w) {
			outs.write(*inb++);
		}
	}
}

void stream_to_array(const int h, const int w, fifo<data_t>& ins, data_ptr_t outb) {
	stream_to_array_hw: for (int i = 0; i < HEIGHT * WIDTH; i++) {
		if (i < h * w) {
			*outb++ = ins.read();
		}
	}
}

void fuse_array2x_to_stream(const int h, const int w, data_ptr_t inbd, data_ptr_t inbt, fifo<data_t>& outs) {
	resize2x_array_to_stream_h: for (int y = 0; y < HEIGHT; y += 2) {
		if (y < h) {
			block_feat_t tmpb;
#pragma HLS bind_storage variable=tmpb type=ram_1p
			resize2x_array_to_stream_w1: for (int x = 0; x < WIDTH; x += 2) {
				if (x < w) {
					tmpb[x / 2] = *inbt++;
					outs.write(*inbd++);
					outs.write(tmpb[x / 2]);
					outs.write(*inbd++);
					outs.write(tmpb[x / 2]);
				}
			}
			resize2x_array_to_stream_w2: for (int x = 0; x < WIDTH; x += 2) {
				if (x < w) {
					outs.write(*inbd++);
					outs.write(tmpb[x / 2]);
					outs.write(*inbd++);
					outs.write(tmpb[x / 2]);
				}
			}
		}
	}
}

void select_array_to_stream(const int h, const int w, const ConvMode& mode,
	block_data_t& inb, block_feat_t& infb, fifo<data_t>& outs)
{
	switch (mode.op) {
	case SpatialOp:
		Conv2D::windowize(HEIGHT * 2, WIDTH * 2, mode.stride, mode.depthwise, inb, outs);
		break;
	case DepthwiseOp:
	case DepthwiseBrOp:
	case DepthwiseHeadOp:
	case SingleChannelOp:
		Conv2D::windowize(h, w, mode.stride, mode.depthwise, inb, outs);
		break;
	case PointwiseOp:
		array_to_stream(h, w, inb, outs);
		break;
	case PointwiseHeadOp:
		array_to_stream(h, w, infb, outs);
		break;
	case FuseTopDownOp:
		fuse_array2x_to_stream(h, w, infb, inb, outs);
		break;
	}
}

void select_stream_to_array(const int h, const int w, const int f, const ConvMode& mode,
	block_data_t& outb, block_feat_t& outfb, fifo<data_t>& ins, pred_t& preds)
{
	switch (mode.op) {
	case SpatialOp:
	case DepthwiseOp:
	case PointwiseOp:
	case PointwiseHeadOp:
	case FuseTopDownOp:
		stream_to_array(h, w, ins, outb);
		break;
	case DepthwiseHeadOp:
		stream_to_array(h, w, ins, outfb);
		break;
	case DepthwiseBrOp:
		branch_to_feature(h, w, ins, outb, outfb);
		break;
	case SingleChannelOp:
		preds.push_raw_pred(h, w, f, ins);
		break;
	}
}

void read_compute_conv(const int h , const int w, const int f, const ConvMode& mode,
	const int nf, const bool concat, fifo<axis_data64>& ins, pred_t& preds,
	const block_conv_t& cur_wi, const block_thr_t& cur_thr,
	block_data_t& inb, block_feat_t& infb, block_data_t& outb, block_feat_t& outfb,
	block_conv_t& next_wi, block_thr_t& next_thr)
{
#pragma HLS inline off
	fifo<data_t> pips1("pipe1_fifo");
	fifo<data_t> pips2("pipe2_fifo");
#pragma HLS stream variable=pips1
#pragma HLS stream variable=pips2

#pragma HLS dataflow

	select_array_to_stream(h, w, mode, inb, infb, pips1);
	Conv2D::compute(h, w, f, mode, cur_wi, cur_thr, pips1, pips2);
	select_stream_to_array(h, w, f, mode, outb, outfb, pips2, preds);
	read_weight(nf, 1, concat, ins, next_wi, next_thr);
}

void maxpool(const int ih, const int iw, const int oh, const int ow,
	block_data_t& iob, block_data_t& tmpb)
{
#pragma HLS inline off
	MaxPool2x2::compute_h(ih, iw, iob, tmpb);
	MaxPool2x2::compute_v(oh, ow, tmpb, iob);
}

void yunet(fifo<axis_data64>& ins, fifo<axis_data8>& outs) {
#pragma HLS interface axis port=ins
#pragma HLS interface axis port=outs
#pragma HLS interface ap_ctrl_none port=return

	static block_data_t buf1;
	static block_data_t buf2;
	static block_conv_t wi1;
	static block_thr_t thr1;
	static block_conv_t wi2;
	static block_thr_t thr2;
#pragma HLS bind_storage variable=buf1 type=ram_1p
#pragma HLS bind_storage variable=buf2 type=ram_1p
#pragma HLS bind_storage variable=wi1 type=ram_1p
#pragma HLS bind_storage variable=wi2 type=ram_1p
#pragma HLS array_partition variable=thr1 cyclic factor=THRESHOLD dim=2
#pragma HLS array_partition variable=thr2 cyclic factor=THRESHOLD dim=2

	static block_feat_t feat8;
	static block_feat_t feat16;
	static block_feat_t feat32;
#pragma HLS bind_storage variable=feat8 type=ram_1p
#pragma HLS bind_storage variable=feat16 type=ram_1p
#pragma HLS bind_storage variable=feat32 type=ram_1p

	pred_t preds;

	// YuNetBackbone stage0
	// Conv_head
	preds.reset();
	read_input(ins, buf2);
	read_weight(16, 3, false, ins, wi1, thr1);
	read_compute_conv(80, 80, 16, Spatial, 16, false, ins, preds, wi1, thr1, buf2, feat16, buf1, feat32, wi2, thr2);
	// Conv_head ConvDPUnit
	read_compute_conv(80, 80, 16, Pointwise, 16, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(80, 80, 16, Depthwise, 16, false, ins, preds, wi1, thr1, buf2, feat16, buf1, feat32, wi2, thr2);
	maxpool(80, 80, 40, 40, buf1, buf2);

	// YuNetBackbone stage1
	// YuNetBackbone Conv4layerBlock 1
	read_compute_conv(40, 40, 16, Pointwise, 16, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(40, 40, 16, Depthwise, 64, false, ins, preds, wi1, thr1, buf2, feat16, buf1, feat32, wi2, thr2);
	// YuNetBackbone Conv4layerBlock 2
	read_compute_conv(40, 40, 64, Pointwise, 64, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(40, 40, 64, Depthwise, 64, true, ins, preds, wi1, thr1, buf2, feat16, buf1, feat32, wi2, thr2);

	// YuNetBackbone stage2
	// YuNetBackbone Conv4layerBlock 1
	read_compute_conv(40, 40, 64, Pointwise, 64, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(40, 40, 64, Depthwise, 64, true, ins, preds, wi1, thr1, buf2, feat16, buf1, feat32, wi2, thr2);
	// YuNetBackbone Conv4layerBlock 2
	read_compute_conv(40, 40, 64, Pointwise, 64, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(40, 40, 64, Depthwise, 64, true, ins, preds, wi1, thr1, buf2, feat16, buf1, feat32, wi2, thr2);
	maxpool(40, 40, 20, 20, buf1, buf2);

	// YuNetBackbone stage3
	// YuNetBackbone Conv4layerBlock 1
	read_compute_conv(20, 20, 64, Pointwise, 64, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(20, 20, 64, Depthwise, 64, true, ins, preds, wi1, thr1, buf2, feat16, buf1, feat32, wi2, thr2);
	// YuNetBackbone Conv4layerBlock 2
	read_compute_conv(20, 20, 64, Pointwise, 64, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(20, 20, 64, DepthwiseBr, 64, true, ins, preds, wi1, thr1, buf2, feat16, buf1, feat8, wi2, thr2);
	maxpool(20, 20, 10, 10, buf1, buf2);

	// YuNetBackbone stage4
	// YuNetBackbone Conv4layerBlock 1
	read_compute_conv(10, 10, 64, Pointwise, 64, false, ins, preds, wi2, thr2, buf1, feat32, buf2, feat8, wi1, thr1);
	read_compute_conv(10, 10, 64, Depthwise, 64, true, ins, preds, wi1, thr1, buf2, feat32, buf1, feat8, wi2, thr2);
	// YuNetBackbone Conv4layerBlock 2
	read_compute_conv(10, 10, 64, Pointwise, 64, false, ins, preds, wi2, thr2, buf1, feat32, buf2, feat8, wi1, thr1);
	read_compute_conv(10, 10, 64, DepthwiseBr, 64, true, ins, preds, wi1, thr1, buf2, feat32, buf1, feat16, wi2, thr2);
	maxpool(10, 10, 5, 5, buf1, buf2);

	// YuNetBackbone stage5
	// YuNetBackbone Conv4layerBlock 1
	read_compute_conv(5, 5, 64, Pointwise, 64, false, ins, preds, wi2, thr2, buf1, feat8, buf2, feat16, wi1, thr1);
	read_compute_conv(5, 5, 64, Depthwise, 64, true, ins, preds, wi1, thr1, buf2, feat8, buf1, feat16, wi2, thr2);
	// YuNetBackbone Conv4layerBlock 2
	read_compute_conv(5, 5, 64, Pointwise, 64, false, ins, preds, wi2, thr2, buf1, feat8, buf2, feat16, wi1, thr1);
	read_compute_conv(5, 5, 64, Depthwise, 64, true, ins, preds, wi1, thr1, buf2, feat8, buf1, feat16, wi2, thr2);

	// TFPN stride32
	// TFPN ConvDPUnit
	read_compute_conv(5, 5, 64, Pointwise, 64, false, ins, preds, wi2, thr2, buf1, feat8, buf2, feat16, wi1, thr1);
	read_compute_conv(5, 5, 64, DepthwiseBr, 64, true, ins, preds, wi1, thr1, buf2, feat8, buf1, feat32, wi2, thr2);
	// TFPN stride16
	// TFPN ConvDPUnit
	read_compute_conv(10, 10, 64, FuseTopDown, 64, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(10, 10, 64, DepthwiseBr, 64, true, ins, preds, wi1, thr1, buf2, feat8, buf1, feat16, wi2, thr2);
	// TFPN stride8
	// TFPN ConvDPUnit
	read_compute_conv(20, 20, 64, FuseTopDown, 64, false, ins, preds, wi2, thr2, buf1, feat8, buf2, feat16, wi1, thr1);
	read_compute_conv(20, 20, 64, DepthwiseBr, 64, true, ins, preds, wi1, thr1, buf2, feat32, buf1, feat8, wi2, thr2);

	// YuNet_Head stride8
	// YuNet_Head shared ConvDPUnit
	read_compute_conv(20, 20, 64, PointwiseHead, 64, false, ins, preds, wi2, thr2, buf1, feat8, buf2, feat16, wi1, thr1);
	read_compute_conv(20, 20, 64, DepthwiseHead, 64, true, ins, preds, wi1, thr1, buf2, feat32, buf1, feat8, wi2, thr2);
	// YuNet_Head stride16
	// YuNet_Head shared ConvDPUnit
	read_compute_conv(10, 10, 64, PointwiseHead, 64, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(10, 10, 64, DepthwiseHead, 64, true, ins, preds, wi1, thr1, buf2, feat8, buf1, feat16, wi2, thr2);
	// YuNet_Head stride32
	// YuNet_Head shared ConvDPUnit
	read_compute_conv(5, 5, 64, PointwiseHead, 64, false, ins, preds, wi2, thr2, buf1, feat32, buf2, feat8, wi1, thr1);
	read_compute_conv(5, 5, 64, DepthwiseHead, 1, true, ins, preds, wi1, thr1, buf2, feat16, buf1, feat32, wi2, thr2);

	// YuNet_Head cls ConvDPUnit
	// YuNet_Head stride8
	read_compute_conv(20, 20, 1, PointwiseHead, 1, false, ins, preds, wi2, thr2, buf1, feat8, buf2, feat16, wi1, thr1);
	read_compute_conv(20, 20, 1, SingleChannel, 1, true, ins, preds, wi1, thr1, buf2, feat32, buf1, feat16, wi2, thr2);
	// YuNet_Head stride16
	read_compute_conv(10, 10, 1, PointwiseHead, 1, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(10, 10, 1, SingleChannel, 1, true, ins, preds, wi1, thr1, buf2, feat8, buf1, feat32, wi2, thr2);
	// YuNet_Head stride32
	read_compute_conv(5, 5, 1, PointwiseHead, 1, false, ins, preds, wi2, thr2, buf1, feat32, buf2, feat8, wi1, thr1);
	read_compute_conv(5, 5, 1, SingleChannel, 4, true, ins, preds, wi1, thr1, buf2, feat16, buf1, feat8, wi2, thr2);

	// YuNet_Head bbox ConvDPUnit
	// YuNet_Head stride8
	read_compute_conv(20, 20, 4, PointwiseHead, 4, false, ins, preds, wi2, thr2, buf1, feat8, buf2, feat16, wi1, thr1);
	read_compute_conv(20, 20, 4, SingleChannel, 4, true, ins, preds, wi1, thr1, buf2, feat32, buf1, feat16, wi2, thr2);
	// YuNet_Head stride16
	read_compute_conv(10, 10, 4, PointwiseHead, 4, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(10, 10, 4, SingleChannel, 4, true, ins, preds, wi1, thr1, buf2, feat8, buf1, feat32, wi2, thr2);
	// YuNet_Head stride32
	read_compute_conv(5, 5, 4, PointwiseHead, 4, false, ins, preds, wi2, thr2, buf1, feat32, buf2, feat8, wi1, thr1);
	read_compute_conv(5, 5, 4, SingleChannel, 1, true, ins, preds, wi1, thr1, buf2, feat16, buf1, feat8, wi2, thr2);

	// YuNet_Head obj ConvDPUnit
	// YuNet_Head stride8
	read_compute_conv(20, 20, 1, PointwiseHead, 1, false, ins, preds, wi2, thr2, buf1, feat8, buf2, feat16, wi1, thr1);
	read_compute_conv(20, 20, 1, SingleChannel, 1, true, ins, preds, wi1, thr1, buf2, feat32, buf1, feat16, wi2, thr2);
	// YuNet_Head stride16
	read_compute_conv(10, 10, 1, PointwiseHead, 1, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(10, 10, 1, SingleChannel, 1, true, ins, preds, wi1, thr1, buf2, feat8, buf1, feat32, wi2, thr2);
	// YuNet_Head stride32
	read_compute_conv(5, 5, 1, PointwiseHead, 1, false, ins, preds, wi2, thr2, buf1, feat32, buf2, feat8, wi1, thr1);
	read_compute_conv(5, 5, 1, SingleChannel, 10, true, ins, preds, wi1, thr1, buf2, feat16, buf1, feat8, wi2, thr2);

	// YuNet_Head kps ConvDPUnit
	// YuNet_Head stride8
	read_compute_conv(20, 20, 10, PointwiseHead, 10, false, ins, preds, wi2, thr2, buf1, feat8, buf2, feat16, wi1, thr1);
	read_compute_conv(20, 20, 10, SingleChannel, 10, true, ins, preds, wi1, thr1, buf2, feat32, buf1, feat16, wi2, thr2);
	// YuNet_Head stride16
	read_compute_conv(10, 10, 10, PointwiseHead, 10, false, ins, preds, wi2, thr2, buf1, feat16, buf2, feat32, wi1, thr1);
	read_compute_conv(10, 10, 10, SingleChannel, 10, true, ins, preds, wi1, thr1, buf2, feat8, buf1, feat32, wi2, thr2);
	// YuNet_Head stride32
	read_compute_conv(5, 5, 10, PointwiseHead, 10, false, ins, preds, wi2, thr2, buf1, feat32, buf2, feat8, wi1, thr1);
	read_compute_conv(5, 5, 10, SingleChannel, 0, true, ins, preds, wi1, thr1, buf2, feat16, buf1, feat8, wi2, thr2);
	
	write_output(preds, outs);
}
