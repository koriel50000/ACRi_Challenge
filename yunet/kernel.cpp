#include "kernel.hpp"
#include "layers.hpp"

void print_data_hist(const int h, const int w, const int c, block_data_t& buf) {
	static const int16_t v0[] = {
		0, 1, 2, 3, 4, 6, 8, 12,
		0, -1, -2, -3, -4, -6, -8, -12,
	};

	int count = 0;
	float sum = 0;
	int hist[16] = {};
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			data_t val = buf[y * WIDTH + x];
			for (int z = 0; z < c; z++) {
				int v = val[z].to_int();
				count++;
				sum += v;
				hist[v]++;
				if (count <= 20) {
					printf("%d ", v0[v]);
				}
			}
		}
	}
	printf("\n");
	printf("mean=%f count=%d\n", sum / count, count);
	for (int i = 15; i > 8; --i) {
		printf("[%d]=%d ", 8 - i, hist[i]);
	}
	for (int i = 0; i < 8; i++) {
		printf("[%d]=%d ", i, hist[i]);
	}
	printf("\n");
}

void print_feature_hist(const int h, const int w, const int c, block_feat_t& buf) {
	static const int16_t v0[] = {
		0, 1, 2, 3, 4, 6, 8, 12,
		0, -1, -2, -3, -4, -6, -8, -12,
	};

	int count = 0;
	float sum = 0;
	int hist[16] = {};
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			data_t val = buf[y * FEAT_WIDTH + x];
			for (int z = 0; z < c; z++) {
				int v = val[z].to_int();
				count++;
				sum += v;
				hist[v]++;
				if (count <= 20) {
					printf("%d ", v0[v]);
				}
			}
		}
	}
	printf("\n");
	printf("mean=%f count=%d\n", sum / count, count);
	for (int i = 15; i > 8; --i) {
		printf("[%d]=%d ", 8 - i, hist[i]);
	}
	for (int i = 0; i < 8; i++) {
		printf("[%d]=%d ", i, hist[i]);
	}
	printf("\n");
}

void print_param_hist(const int f, const int kn, const int c, block_conv_t& wi, block_thr_t& thr) {
	static const int16_t v0[] = {
		0, 1, 2, 4, 8, 16, 32, 64,
		0, -1, -2, -4, -8, -16, -32, -64,
	};

	int count = 0;
	float sum = 0;
	int hist[16] = {};
	for (int i = 0; i < f * kn * kn; i++) {
		data_t val = wi[i];
		for (int z = 0; z < c; z++) {
			int v = val[z].to_int();
			count++;
			sum += v;
			hist[v]++;
			if (count <= 20) {
				printf("%d ", v0[v]);
			}
		}
	}

	printf("\n");
	printf("mean=%f count=%d\n", sum / count, count);
	for (int i = 15; i > 8; --i) {
		printf("[%d]=%d ", 8 - i, hist[i]);
	}
	for (int i = 0; i < 8; i++) {
		printf("[%d]=%d ", i, hist[i]);
	}
	printf("\n");

	for (int j = 0; j < f; j++) {
		printf("[%d] = { ", j);
		for (int i = 0; i < THRESHOLD; i++) {
			printf("%d, ", thr[j][i]);
		}
		printf("}\n");
	}
	printf("\n");
}

template <bool CONCAT = false>
void read_weight(const int f, const int kn,
	fifo<uint64_t>& ins, block_conv_t& outw, block_thr_t& outh)
{
	read_weight_f: for (int i = 0; i < FILTER * KERNEL * KERNEL; i++) {
#pragma HLS pipeline
		if (i < f * kn * kn) {
			if (CONCAT) {
				uint64_t w3 = ins.read();
				uint64_t w2 = ins.read();
				uint64_t w1 = ins.read();
				uint64_t w0 = ins.read();
				outw[i] = data_t(w3, w2, w1, w0);
			} else {
				outw[i] = data_t(ins.read());
			}
		}
	}

	read_threshold_f: for (int j = 0; j < FILTER; j++) {
		if (j < f) {
			read_threshold_t: for (int i = 0; i < THRESHOLD; i++) {
	#pragma HLS pipeline
				outh[j][i] = ins.read();
			}
		}
	}
}

Conv2Dstride<HEIGHT*2,WIDTH*2,CHANNEL,FILTER,KERNEL,2> conv3x3_stride;
Conv2Dpointwise<HEIGHT,WIDTH,CHANNEL,FILTER> conv1x1;
Conv2Ddepthwise<HEIGHT,WIDTH,CHANNEL,FILTER,KERNEL> conv3x3_depthwise;
MaxPool2x2<HEIGHT,WIDTH,CHANNEL> maxpool2x2;

void input_compute_conv3x3_stride(
	const int h, const int w, const int f, fifo<uint64_t>& ins, linebuf_t& linebuf,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& buf)
{
	fifo<win_t> pips("pipe_fifo");

#pragma HLS dataflow
	conv3x3_stride.windowize(linebuf, ins, pips);
	conv3x3_stride.compute(h, w, f, cur_wi, cur_thr, pips, buf);
}

void read_compute_conv1x1(
	const int h, const int w, const int f,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& buf,
	const int nf, const int nkn,
	fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
#pragma HLS dataflow
	conv1x1.compute(h, w, f, cur_wi, cur_thr, buf);
	read_weight(nf, nkn, ins, next_wi, next_thr);
}

void read_compute_conv1x1(
	const int h, const int w, const int f,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_feat_t& infb, block_data_t& outb,
	const int nf, const int nkn,
	fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
#pragma HLS dataflow
	conv1x1.compute(h, w, f, cur_wi, cur_thr, infb, outb);
	read_weight(nf, nkn, ins, next_wi, next_thr);
}

template <bool CONCAT = false>
void read_compute_conv3x3dw_relu(
	const int h, const int w, const int f, linebuf_t& linebuf,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& buf,
	const int nf, const int nkn,
	fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
	fifo<win_t> pips("pipe_fifo");

#pragma HLS dataflow
	conv3x3_depthwise.windowize(h, w, linebuf, buf, pips);
	conv3x3_depthwise.compute(h, w, f, true, cur_wi, cur_thr, pips, buf);
	read_weight<CONCAT>(nf, nkn, ins, next_wi, next_thr);
}

template <bool CONCAT = false>
void compute_branch_conv3x3dw_relu(
	const int h, const int w, const int f, linebuf_t& linebuf,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& buf, block_feat_t& outfb,
	const int nf, const int nkn,
	fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
	fifo<win_t> pips("pipe_fifo");

#pragma HLS dataflow
	conv3x3_depthwise.windowize(h, w, linebuf, buf, pips);
	conv3x3_depthwise.compute_branch(h, w, f, cur_wi, cur_thr, pips, buf, outfb);
	read_weight<CONCAT>(nf, nkn, ins, next_wi, next_thr);
}

void fuse_topdown_compute_conv1x1(
	const int h, const int w, const int f,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_feat_t& infb, block_data_t& buf,
	const int nf, const int nkn,
	fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
#pragma HLS dataflow
	conv1x1.fuse_topdown_compute(h, w, f, cur_wi, cur_thr, infb, buf);
	read_weight(nf, nkn, ins, next_wi, next_thr);
}

void read_compute_conv3x3sc(
	const int h, const int w, const int f, linebuf_t& linebuf,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& buf,
	const int nf, const int nkn,
	fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
	fifo<win_t> pips("pipe_fifo");

#pragma HLS dataflow
	conv3x3_depthwise.windowize(h, w, linebuf, buf, pips);
	conv3x3_depthwise.compute(h, w, f, false, cur_wi, cur_thr, pips, buf);
	read_weight<true>(nf, nkn, ins, next_wi, next_thr);
}

void output_compute_conv3x3sc(
	const int h, const int w, const int f, linebuf_t& linebuf,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& buf)
{
	fifo<win_t> pips("pipe_fifo");

#pragma HLS dataflow
	conv3x3_depthwise.windowize(h, w, linebuf, buf, pips);
	conv3x3_depthwise.compute(h, w, f, false, cur_wi, cur_thr, pips, buf);
}

void compute_maxpool2x2(const int h, const int w, block_data_t& buf) {
	fifo<data_t> pips("pipe_fifo");

#pragma HLS dataflow
	maxpool2x2.compute_h(h, w, buf, pips);
	maxpool2x2.compute_v(h / 2, w / 2, pips, buf);
}

void kernel(fifo<uint64_t>& ins, int out[16]) {
#pragma HLS interface axis port=ins
#pragma HLS interface axis port=out

	static block_data_t stage_buf;
	static block_conv_t even_wi;
	static block_thr_t even_thr;
	static block_conv_t odd_wi;
	static block_thr_t odd_thr;
#pragma HLS bind_storage variable=stage_buf type=ram_2p impl=bram
#pragma HLS bind_storage variable=even_wi type=ram_1p impl=bram
#pragma HLS bind_storage variable=even_thr type=ram_1p impl=bram
#pragma HLS bind_storage variable=odd_wi type=ram_1p impl=bram
#pragma HLS bind_storage variable=odd_thr type=ram_1p impl=bram

	static block_feat_t feat8_buf;
	static block_feat_t feat16_buf;
	static block_feat_t feat32_buf;
#pragma HLS bind_storage variable=feat8_buf type=ram_1p impl=bram
#pragma HLS bind_storage variable=feat16_buf type=ram_1p impl=bram
#pragma HLS bind_storage variable=feat32_buf type=ram_1p impl=bram

	linebuf_t linebuf;

	read_weight(16, 3, ins, even_wi, even_thr);
	read_weight(16, 1, ins, odd_wi, odd_thr);
	// YuNetBackbone stage0
	// Conv_head
	input_compute_conv3x3_stride(
		80, 80, 16, ins, linebuf, even_wi, even_thr, stage_buf);
	// Conv_head ConvDPUnit
	read_compute_conv1x1(
		80, 80, 16, odd_wi, odd_thr, stage_buf,
		16, 1, ins, even_wi, even_thr);
	read_compute_conv3x3dw_relu(
		80, 80, 16, linebuf, even_wi, even_thr, stage_buf,
		16, 1, ins, odd_wi, odd_thr);
	compute_maxpool2x2(80, 80, stage_buf);

	// YuNetBackbone stage1
	// YuNetBackbone Conv4layerBlock 1
	read_compute_conv1x1(
		40, 40, 16, odd_wi, odd_thr, stage_buf,
		16, 1, ins, even_wi, even_thr);
	read_compute_conv3x3dw_relu(
		40, 40, 16, linebuf, even_wi, even_thr, stage_buf,
		64, 1, ins, odd_wi, odd_thr);
	// YuNetBackbone Conv4layerBlock 2
	read_compute_conv1x1(
		40, 40, 64, odd_wi, odd_thr, stage_buf,
		64, 1, ins, even_wi, even_thr);
	read_compute_conv3x3dw_relu<true>(
		40, 40, 64, linebuf, even_wi, even_thr, stage_buf,
		64, 1, ins, odd_wi, odd_thr);

	// YuNetBackbone stage2
	// YuNetBackbone Conv4layerBlock 1
	read_compute_conv1x1(
		40, 40, 64, odd_wi, odd_thr, stage_buf,
		64, 1, ins, even_wi, even_thr);
	read_compute_conv3x3dw_relu<true>(
		40, 40, 64, linebuf, even_wi, even_thr, stage_buf,
		64, 1, ins, odd_wi, odd_thr);
	// YuNetBackbone Conv4layerBlock 2
	read_compute_conv1x1(
		40, 40, 64, odd_wi, odd_thr, stage_buf,
		64, 1, ins, even_wi, even_thr);
	read_compute_conv3x3dw_relu<true>(
		40, 40, 64, linebuf, even_wi, even_thr, stage_buf,
		64, 1, ins, odd_wi, odd_thr);
	compute_maxpool2x2(40, 40, stage_buf);

	// YuNetBackbone stage3
	// YuNetBackbone Conv4layerBlock 1
	read_compute_conv1x1(
		20, 20, 64, odd_wi, odd_thr, stage_buf,
		64, 1, ins, even_wi, even_thr);
	read_compute_conv3x3dw_relu<true>(
		20, 20, 64, linebuf, even_wi, even_thr, stage_buf,
		64, 1, ins, odd_wi, odd_thr);
	// YuNetBackbone Conv4layerBlock 2
	read_compute_conv1x1(
		20, 20, 64, odd_wi, odd_thr, stage_buf,
		64, 1, ins, even_wi, even_thr);
	compute_branch_conv3x3dw_relu<true>(
		20, 20, 64, linebuf, even_wi, even_thr, stage_buf, feat8_buf,
		64, 1, ins, odd_wi, odd_thr);
	compute_maxpool2x2(20, 20, stage_buf);

	// YuNetBackbone stage4
	// YuNetBackbone Conv4layerBlock 1
	read_compute_conv1x1(
		10, 10, 64, odd_wi, odd_thr, stage_buf,
		64, 1, ins, even_wi, even_thr);
	read_compute_conv3x3dw_relu<true>(
		10, 10, 64, linebuf, even_wi, even_thr, stage_buf,
		64, 1, ins, odd_wi, odd_thr);
	// YuNetBackbone Conv4layerBlock 2
	read_compute_conv1x1(
		10, 10, 64, odd_wi, odd_thr, stage_buf,
		64, 1, ins, even_wi, even_thr);
	compute_branch_conv3x3dw_relu<true>(
		10, 10, 64, linebuf, even_wi, even_thr, stage_buf, feat16_buf,
		64, 1, ins, odd_wi, odd_thr);
	compute_maxpool2x2(10, 10, stage_buf);

	// YuNetBackbone stage5
	// YuNetBackbone Conv4layerBlock 1
	read_compute_conv1x1(
		5, 5, 64, odd_wi, odd_thr, stage_buf,
		64, 1, ins, even_wi, even_thr);
	read_compute_conv3x3dw_relu<true>(
		5, 5, 64, linebuf, even_wi, even_thr, stage_buf,
		64, 1, ins, odd_wi, odd_thr);
	// YuNetBackbone Conv4layerBlock 2
	read_compute_conv1x1(
		5, 5, 64, odd_wi, odd_thr, stage_buf,
		64, 1, ins, even_wi, even_thr);
	read_compute_conv3x3dw_relu<true>(
		5, 5, 64, linebuf, even_wi, even_thr, stage_buf,
		64, 1, ins, odd_wi, odd_thr);

	// TFPN stride32
	// TFPN ConvDPUnit
	read_compute_conv1x1(
		5, 5, 64, odd_wi, odd_thr, stage_buf,
		64, 1, ins, even_wi, even_thr);
	compute_branch_conv3x3dw_relu<true>(
		5, 5, 64, linebuf, even_wi, even_thr, stage_buf, feat32_buf,
		64, 1, ins, odd_wi, odd_thr);
	// TFPN stride16
	// TFPN ConvDPUnit
	fuse_topdown_compute_conv1x1(
		10, 10, 64, odd_wi, odd_thr, feat16_buf, stage_buf,
		64, 1, ins, even_wi, even_thr);
	compute_branch_conv3x3dw_relu<true>(
		10, 10, 64, linebuf, even_wi, even_thr, stage_buf, feat16_buf,
		64, 1, ins, odd_wi, odd_thr);
	// TFPN stride8
	// TFPN ConvDPUnit
	fuse_topdown_compute_conv1x1(
		20, 20, 64, odd_wi, odd_thr, feat8_buf, stage_buf,
		64, 1, ins, even_wi, even_thr);
	read_compute_conv3x3dw_relu<true>(
		20, 20, 64, linebuf, even_wi, even_thr, stage_buf,
		64, 1, ins, odd_wi, odd_thr);

	// YuNet_Head stride8
	// YuNet_Head shared ConvDPUnit
	read_compute_conv1x1(
		20, 20, 64, odd_wi, odd_thr, stage_buf,
		64, 1, ins, even_wi, even_thr);
	compute_branch_conv3x3dw_relu<true>(
		20, 20, 64, linebuf, even_wi, even_thr, stage_buf, feat8_buf,
		64, 1, ins, odd_wi, odd_thr);
	// YuNet_Head stride16
	// YuNet_Head shared ConvDPUnit
	read_compute_conv1x1(
		10, 10, 64, odd_wi, odd_thr, feat16_buf, stage_buf,
		64, 1, ins, even_wi, even_thr);
	compute_branch_conv3x3dw_relu<true>(
		10, 10, 64, linebuf, even_wi, even_thr, stage_buf, feat16_buf,
		64, 1, ins, odd_wi, odd_thr);
	// YuNet_Head stride32
	// YuNet_Head shared ConvDPUnit
	read_compute_conv1x1(
		5, 5, 64, odd_wi, odd_thr, feat32_buf, stage_buf,
		64, 1, ins, even_wi, even_thr);
	compute_branch_conv3x3dw_relu<true>(
		5, 5, 64, linebuf, even_wi, even_thr, stage_buf, feat32_buf,
		1, 1, ins, odd_wi, odd_thr);

	// YuNet_Head cls ConvDPUnit
	// YuNet_Head stride8
	read_compute_conv1x1(
		20, 20, 1, odd_wi, odd_thr, feat8_buf, stage_buf,
		1, 1, ins, even_wi, even_thr);
	read_compute_conv3x3sc(
		20, 20, 1, linebuf, even_wi, even_thr, stage_buf,
		1, 1, ins, odd_wi, odd_thr);
//	print_data_hist(20, 20, 1, stage_buf);
	// YuNet_Head stride16
	read_compute_conv1x1(
		10, 10, 1, odd_wi, odd_thr, feat16_buf, stage_buf,
		1, 1, ins, even_wi, even_thr);
	read_compute_conv3x3sc(
		10, 10, 1, linebuf, even_wi, even_thr, stage_buf,
		1, 1, ins, odd_wi, odd_thr);
//	print_data_hist(10, 10, 1, stage_buf);
	// YuNet_Head stride32
	read_compute_conv1x1(
		5, 5, 1, odd_wi, odd_thr, feat32_buf, stage_buf,
		1, 1, ins, even_wi, even_thr);
	read_compute_conv3x3sc(
		5, 5, 1, linebuf, even_wi, even_thr, stage_buf,
		4, 1, ins, odd_wi, odd_thr);
//	print_data_hist(5, 5, 1, stage_buf);

	// YuNet_Head bbox ConvDPUnit
	// YuNet_Head stride8
	read_compute_conv1x1(
		20, 20, 4, odd_wi, odd_thr, feat8_buf, stage_buf,
		4, 1, ins, even_wi, even_thr);
	read_compute_conv3x3sc(
		20, 20, 4, linebuf, even_wi, even_thr, stage_buf,
		4, 1, ins, odd_wi, odd_thr);
//	print_data_hist(20, 20, 4, stage_buf);
	// YuNet_Head stride16
	read_compute_conv1x1(
		10, 10, 4, odd_wi, odd_thr, feat16_buf, stage_buf,
		4, 1, ins, even_wi, even_thr);
	read_compute_conv3x3sc(
		10, 10, 4, linebuf, even_wi, even_thr, stage_buf,
		4, 1, ins, odd_wi, odd_thr);
//	print_data_hist(10, 10, 4, stage_buf);
	// YuNet_Head stride32
	read_compute_conv1x1(
		5, 5, 4, odd_wi, odd_thr, feat32_buf, stage_buf,
		4, 1, ins, even_wi, even_thr);
	read_compute_conv3x3sc(
		5, 5, 4, linebuf, even_wi, even_thr, stage_buf,
		1, 1, ins, odd_wi, odd_thr);
//	print_data_hist(5, 5, 4, stage_buf);

	// YuNet_Head obj ConvDPUnit
	// YuNet_Head stride8
	read_compute_conv1x1(
		20, 20, 1, odd_wi, odd_thr, feat8_buf, stage_buf,
		1, 1, ins, even_wi, even_thr);
	read_compute_conv3x3sc(
		20, 20, 1, linebuf, even_wi, even_thr, stage_buf,
		1, 1, ins, odd_wi, odd_thr);
//	print_data_hist(20, 20, 1, stage_buf);
	// YuNet_Head stride16
	read_compute_conv1x1(
		10, 10, 1, odd_wi, odd_thr, feat16_buf, stage_buf,
		1, 1, ins, even_wi, even_thr);
	read_compute_conv3x3sc(
		10, 10, 1, linebuf, even_wi, even_thr, stage_buf,
		1, 1, ins, odd_wi, odd_thr);
//	print_data_hist(10, 10, 1, stage_buf);
	// YuNet_Head stride32
	read_compute_conv1x1(
		5, 5, 1, odd_wi, odd_thr, feat32_buf, stage_buf,
		1, 1, ins, even_wi, even_thr);
	read_compute_conv3x3sc(
		5, 5, 1, linebuf, even_wi, even_thr, stage_buf,
		10, 1, ins, odd_wi, odd_thr);
//	print_data_hist(5, 5, 1, stage_buf);

	// YuNet_Head kps ConvDPUnit
	// YuNet_Head stride8
	read_compute_conv1x1(
		20, 20, 10, odd_wi, odd_thr, feat8_buf, stage_buf,
		10, 1, ins, even_wi, even_thr);
	read_compute_conv3x3sc(
		20, 20, 10, linebuf, even_wi, even_thr, stage_buf,
		10, 1, ins, odd_wi, odd_thr);
//	print_data_hist(20, 20, 10, stage_buf);
	// YuNet_Head stride16
	read_compute_conv1x1(
		10, 10, 10, odd_wi, odd_thr, feat16_buf, stage_buf,
		10, 1, ins, even_wi, even_thr);
	read_compute_conv3x3sc(
		10, 10, 10, linebuf, even_wi, even_thr, stage_buf,
		10, 1, ins, odd_wi, odd_thr);
//	print_data_hist(10, 10, 10, stage_buf);
	// YuNet_Head stride32
	read_compute_conv1x1(
		5, 5, 10, odd_wi, odd_thr, feat32_buf, stage_buf,
		10, 1, ins, even_wi, even_thr);
	output_compute_conv3x3sc(
		5, 5, 10, linebuf, even_wi, even_thr, stage_buf);
//	print_data_hist(5, 5, 10, stage_buf);
	for (int i = 0; i < 16; i++) {
		out[i] = 0;
	}
}
