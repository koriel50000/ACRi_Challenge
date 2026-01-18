#include "kernel.hpp"
#include "layers.hpp"

void read_data(const int h, const int w, const int c,
	fifo<uint64_t>& ins, block_data_t& outb)
{
	for (int y = 0; y < HEIGHT; y++) {
		if (y >= h) break;
		for (int x = 0; x < WIDTH; x++) {
			if (x >= w) break;
			data_t val = data_t(ins.read());
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

void read_compute_conv3x3_stride(
	const int h, const int w, const int c, const int f, linebuf_t& linebuf,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& inb, block_data_t& outb,
	const int nf, const int nkn,
	fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
	Conv2D<HEIGHT,WIDTH,CHANNEL,FILTER,KERNEL,true,2> conv3x3_stride;
	fifo<win_t> pips1("pipe_fifo1");

#pragma HLS dataflow
	conv3x3_stride.windowize(h, w, linebuf, inb, pips1);
	conv3x3_stride.compute(h / 2, w / 2, c, f, cur_wi, cur_thr, pips1, outb);
	read_weight(nf, nkn, ins, next_wi, next_thr);
}

void read_compute_conv1x1(
	const int h, const int w, const int c, const int f,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& inb, block_data_t& outb,
	const int nf, const int nkn,
    fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
	Conv2D1x1<HEIGHT,WIDTH,CHANNEL,FILTER> conv1x1;

#pragma HLS dataflow
	conv1x1.compute(h, w, c, f, cur_wi, cur_thr, inb, outb);
	read_weight(nf, nkn, ins, next_wi, next_thr);
}

void read_compute_conv3x3_relu(
	const int h, const int w, const int c, const int f, linebuf_t& linebuf,
	block_conv_t& cur_wi, block_thr_t& cur_thr, block_data_t& inb, block_data_t& outb,
	const int nf, const int nkn,
	fifo<uint64_t>& ins, block_conv_t& next_wi, block_thr_t& next_thr)
{
	Conv2D<HEIGHT,WIDTH,CHANNEL,FILTER,3,true> conv3x3;
	fifo<win_t> pips1("pipe_fifo1");

#pragma HLS dataflow
	conv3x3.windowize(h, w, linebuf, inb, pips1);
	conv3x3.compute(h, w, c, f, cur_wi, cur_thr, pips1, outb);
	read_weight(nf, nkn, ins, next_wi, next_thr);
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

void kernel(fifo<uint64_t>& ins, int out[16]) {
#pragma HLS interface axis port=ins
#pragma HLS interface axis port=out

	static block_data_t even_buf;
	static block_data_t odd_buf;
	static block_conv_t even_wi;
	static block_thr_t even_thr;
	static block_conv_t odd_wi;
	static block_thr_t odd_thr;
#pragma HLS bind_storage variable=even_buf type=ram_1p impl=bram
#pragma HLS bind_storage variable=odd_buf type=ram_1p impl=bram
#pragma HLS bind_storage variable=even_wi type=ram_1p impl=bram
#pragma HLS bind_storage variable=even_thr type=ram_1p impl=bram
#pragma HLS bind_storage variable=odd_wi type=ram_1p impl=bram
#pragma HLS bind_storage variable=odd_thr type=ram_1p impl=bram

	linebuf_t linebuf;

	read_data(160, 160, 3, ins, even_buf);
	read_weight(16, 3, ins, even_wi, even_thr);
	// Conv_head
	read_compute_conv3x3_stride(
		160, 160, 3, 16, linebuf, even_wi, even_thr, even_buf, odd_buf,
		16, 1, ins, odd_wi, odd_thr);
	// Conv_head ConvDPUnit
	read_compute_conv1x1(
		80, 80, 16, 16, odd_wi, odd_thr, odd_buf, even_buf,
		16, 3, ins, even_wi, even_thr);
	read_compute_conv3x3_relu(
		80, 80, 16, 16, linebuf, even_wi, even_thr, even_buf, odd_buf,
		16, 1, ins, odd_wi, odd_thr);
	// YuNetBackbone
	compute_maxpool2x2(80, 80, 16, odd_buf, even_buf);
print_data_hist(40, 40, 16, odd_buf);
	// YuNetBackbone Conv4layerBlock 1
	//read_compute_conv1x1(40, 40, 16, 1, 16, 1, 3,
	//    ins, even_wi, even_thr, odd_wi, odd_thr, even_buf, odd_buf);
	//read_compute_conv3x3_relu(40, 40, 16, 1, 64, 16, 1,
	//    ins, odd_wi, odd_thr, even_wi, even_thr, odd_buf, even_buf);
	// YuNetBackbone Conv4layerBlock 2
	//read_compute_conv1x1(40, 40, 16, 1, 64, 1, 3,
	//    ins, even_wi, even_thr, odd_wi, odd_thr, even_buf, odd_buf);
	//read_compute_conv3x3_relu(40, 40, 16, 64, 64, 64, 1,
	//    ins, odd_wi, odd_thr, even_wi, even_thr, odd_buf, even_buf);
//	// YuNetBackbone Conv4layerBlock 3
//	read_compute_conv1x1(40, 40, 16, 64, 64, 1, 3,
//	    ins, even_wi, even_thr, odd_wi, odd_thr, even_buf, odd_buf);
//	read_compute_conv3x3_relu(40, 40, 16, 64, 64, 64, 1,
//	    ins, odd_wi, odd_thr, even_wi, even_thr, odd_buf, even_buf);
//	// YuNetBackbone Conv4layerBlock 4
//	read_compute_conv1x1(40, 40, 64, 64, 64, 1, 3,
//	    ins, even_wi, even_thr, odd_wi, odd_thr, even_buf, odd_buf);
//	read_compute_conv3x3_relu(40, 40, 64, 64, 64, 64, 1,
//	    ins, odd_wi, odd_thr, even_wi, even_thr, odd_buf, even_buf);

	//compute_conv2d_1x1<16, 1>(buf16f, buf1b,
	//	(int_t<4,16>**)backbone_model1_conv1_conv1_weight, // [16][1]
	//	(int**)backbone_model1_conv1_quant1_threshold, false, // [16][14]
	//	80, 80);
	//compute_conv2d<1, 16>(buf1b, buf16f,
	//	(int_t<4,1>**)backbone_model1_conv1_conv2_weight, // [16][9]
	//	(int**)backbone_model1_conv1_relu2_threshold, true, // [16][7]
	//	80, 80, 80, 80, 3, 1);

	//write_result<80, 80, 16>(out, buf16f);

	// fifo<int_t<4,4>> ins("input_fifo");
	// fifo<win_t<int_t<4,4>,3*3>> pips1("pipe_fifo1");
	// fifo<int_t<4,16>> pips2("pipe_fifo2");
	// fifo<int_t<4,1>> pips3("pipe_fifo3");
	// fifo<win_t<int_t<4,1>,3*3>> pips4("pipe_fifo4");
	// fifo<int_t<4,16>> pips5("pipe_fifo5");
	// fifo<int_t<4,16>> pips6("pipe_fifo6");
	// fifo<int_t<4,16>> pips7("pipe_fifo7");

	// fifo<int_t<4,1>> pips8("pipe_fifo8");
	// fifo<win_t<int_t<4,1>,3*3>> pips9("pipe_fifo9");
	// fifo<int_t<4,16>> pips10("pipe_fifo10");

	// Conv2D<320,320,4,3,1,2> backbone_model0_conv1;
	// Conv2D<160,160,16,1> backbone_model0_conv2_1;
	// Conv2D<160,160,1,3,1> backbone_model0_conv2_2;
	// MaxPool2x2<160, 160, 16> backbone_model0_maxpool3;

	// Conv2D<80,80,16,1> backbone_model1_conv1_1;
	// Conv2D<80,80,1,3,1> backbone_model1_conv1_2;


	// backbone_model0_conv1.windowize(ins, pips1);
	// backbone_model0_conv1.compute<160,160,16,7>(pips1, pips2,
	// backbone_model0_conv2_1.compute<160,160,1,14>(pips2, pips3,
	// backbone_model0_conv2_2.windowize(pips3, pips4);
	// backbone_model0_conv2_2.compute<160,160,16,7>(pips4, pips5,
	// backbone_model0_maxpool3.compute_h(pips5, pips6);
	// backbone_model0_maxpool3.compute_v(pips6, pips7);

	// backbone_model1_conv1_1.compute<80,80,1,14>(pips7, pips8,
	// backbone_model1_conv1_2.windowize(pips8, pips9);
	// backbone_model1_conv1_2.compute<80,80,16,7>(pips9, pips10,
	for (int i = 0; i < 16; i++) {
		out[i] = 0;
	}
}
