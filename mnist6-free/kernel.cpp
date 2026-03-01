#include "kernel.hpp"
#include "layers.hpp"

void read_input(fifo<uint64_t>& ins, block_data_t& outb) {
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
			data_t val = data_t(12 - ins.read() * 8);
			outb[y * WIDTH + x] = val;
		}
	}
}

void read_weight(const int f, const int kn,
	fifo<uint64_t>& ins, block_conv_t& outw, block_thr_t& outh)
{
	for (int i = 0; i < FILTER * KERNEL * KERNEL; i++) {
		if (i >= f * kn * kn) break;
		outw[i] = data_t(ins.read());
	}

	for (int j = 0; j < FILTER; j++) {
		if (j >= f) break;
		for (int i = 0; i < THRESHOLD; i++) {
			outh[j][i] = ins.read();
		}
	}
}

void read_mat_weight(fifo<uint64_t>& ins, block_mat_t& mat_wi) {
	for (int i = 0; i < CLASS * FLATTEN / CHUNK_SIZE; i++) {
		mat_wi[i] = data_t(ins.read());
	}
}

void read_compute_conv3x3_stride(
	const int h, const int w, const int f, linebuf_t& linebuf,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& inb, block_data_t& outb,
	const int nf, const int nkn,
	fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
	Conv2D<HEIGHT,WIDTH,CHANNEL,FILTER,KERNEL,true,2> conv3x3_stride;
	fifo<win_t> pips1("pipe_fifo1");

#pragma HLS dataflow
	conv3x3_stride.windowize(h, w, linebuf, inb, pips1);
	conv3x3_stride.compute(h / 2, w / 2, f, cur_wi, cur_thr, pips1, outb);
	read_weight(nf, nkn, ins, next_wi, next_thr);
}

void read_compute_conv1x1(
	const int h, const int w, const int f,
	block_conv_t& cur_wi, block_thr_t& cur_thr,
	block_data_t& inb, block_data_t& outb,
	const int nf, const int nkn,
	fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
	Conv2Dpointwise<HEIGHT,WIDTH,CHANNEL,FILTER> conv1x1;

#pragma HLS dataflow
	conv1x1.compute(h, w, f, cur_wi, cur_thr, inb, outb);
	read_weight(nf, nkn, ins, next_wi, next_thr);
}

void read_compute_conv3x3dw_relu(
	const int h, const int w, const int f, linebuf_t& linebuf,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& inb, block_data_t& outb,
	fifo<uint64_t>& ins, block_mat_t& mat_wi)
{
	Conv2Ddepthwise<HEIGHT,WIDTH,CHANNEL,FILTER,KERNEL,true> conv3x3_depthwise;
	fifo<win_t> pips1("pipe_fifo1");

#pragma HLS dataflow
	conv3x3_depthwise.windowize(h, w, linebuf, inb, pips1);
	conv3x3_depthwise.compute(h, w, f, cur_wi, cur_thr, pips1, outb);
	read_mat_weight(ins, mat_wi);
}

void compute_maxpool2x2(const int h, const int w, const int c,
	block_data_t& inb, block_data_t& outb)
{
	MaxPool2x2<HEIGHT,WIDTH,CHANNEL> maxpool2x2;
	fifo<data_t> pips1("pipe_fifo1");

#pragma HLS dataflow
	maxpool2x2.compute_h(h, w, c, inb, pips1);
	maxpool2x2.compute_v(h / 2, w / 2, c, pips1, outb);
}

void compute_matmul_write(block_mat_t& mat_wi, block_data_t& inb, int out[1]) {
	Dense<CLASS,FLATTEN,CHUNK_SIZE,7,7> matmul;
	fifo<int_t<CLASS,16>> pips("pipe_fifo");  // int16_t

#pragma HLS dataflow
	matmul.flatten(mat_wi, inb, pips);
	matmul.write_result(out, pips);
}

void print_data_hist(const int h, const int w, const int c, block_data_t& buf) {
	int count = 0;
	float sum = 0;
	int hist[15] = {};
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			data_t val = buf[y * WIDTH + x];
			for (int z = 0; z < c; z++) {
				int v = val[z].to_int();
				count++;
				sum += v;
				hist[v]++;
				if (count <= 20) {
					printf("%d ", v);
				}
			}
		}
	}
	printf("\n");
	printf("mean=%f\n", sum / count);
	for (int i = 15; i > 8; --i) {
		printf("[%d]=%d ", 8 - i, hist[i]);
	}
	for (int i = 0; i < 8; i++) {
		printf("[%d]=%d ", i, hist[i]);
	}
	printf("\n");
}

void kernel(fifo<uint64_t>& ins, int out[1]) {
#pragma HLS interface axis port=ins
#pragma HLS interface axis port=out

	static block_data_t even_buf;
	static block_data_t odd_buf;
	static block_conv_t even_wi;
	static block_thr_t even_thr;
	static block_conv_t odd_wi;
	static block_thr_t odd_thr;
	static block_mat_t mat_wi;
#pragma HLS bind_storage variable=even_buf type=ram_2p impl=bram
#pragma HLS bind_storage variable=odd_buf type=ram_2p impl=bram
#pragma HLS bind_storage variable=even_wi type=ram_1p impl=bram
#pragma HLS bind_storage variable=even_thr type=ram_1p impl=bram
#pragma HLS bind_storage variable=odd_wi type=ram_1p impl=bram
#pragma HLS bind_storage variable=odd_thr type=ram_1p impl=bram
#pragma HLS bind_storage variable=mat_wi type=ram_1p impl=bram

	static linebuf_t linebuf;

	read_input(ins, even_buf);
	read_weight(16, 3, ins, even_wi, even_thr);
	// Conv_head
	read_compute_conv3x3_stride(
		28, 28, 16, linebuf, even_wi, even_thr, even_buf, odd_buf,
		16, 1, ins, odd_wi, odd_thr);
	// Conv_head ConvDPUnit
	read_compute_conv1x1(
		14, 14, 16, odd_wi, odd_thr, odd_buf, even_buf,
		16, 1, ins, even_wi, even_thr);
	read_compute_conv3x3dw_relu(
		14, 14, 16, linebuf, even_wi, even_thr, even_buf, odd_buf,
		ins, mat_wi);
	// YuNetBackbone
	compute_maxpool2x2(14, 14, 16, odd_buf, even_buf);
	compute_matmul_write(mat_wi, even_buf, out);
}
