#include "kernel.hpp"
#include <hls_stream.h>
#include <ap_int.h>
#include <hls_vector.h>
#include <hls_math.h>
//#include <assert.h>
//#include <multimediaIps/xf_video_mem.hpp>

const int WIDTH = 12;
const int HEIGHT = 12;
const int CHANNEL = 16;

const int FILTER = 16;
const int KERNEL = 5;
const int THRESHOLD = 3;

const int OWIDTH = 4;
const int OHEIGHT = 4;

typedef ap_uint<4> uint4_t;
typedef ap_uint<CHANNEL * 4> cpack_t;
typedef hls::vector<uint64_t, KERNEL * KERNEL> wpack_t;
typedef hls::stream<wpack_t> wfifo_t;

typedef ap_uint<2> uint2_t;
typedef ap_uint<FILTER * 2> mpack_t;
typedef hls::stream<mpack_t> mfifo_t;

template <int ROWS, int COLS, typename T>
class LineBuffer {
private:
	hls::vector<T, ROWS * COLS> buf;
public:
	void shift_pixels_up(int col) {
		for (int i = 0; i < ROWS * COLS - 1; i++) {
#pragma HLS unroll
			buf[i] = buf[i + 1];
		}
	}

	void insert_bottom_row(T value, int col) {
		buf[ROWS * COLS - 1] = value;
	}

	void get_col(T value[ROWS], int col) {
		for (int i = 0; i < ROWS; i++) {
#pragma HLS unroll
			value[i] = buf[i * COLS];
		}
	}
};

template <int ROWS, int COLS, typename T>
class Window {
private:
	hls::vector<T, ROWS * COLS> buf;
public:
	void shift_pixels_left() {
		for (int i = 0; i < ROWS * COLS - 1; i++) {
#pragma HLS unroll
			buf[i] = buf[i + 1];
		}
	}

	void insert_right_col(T value[ROWS]) {
		for (int i = 0; i < ROWS; i++) {
#pragma HLS unroll
			buf[(i + 1) * COLS - 1] = value[i];
		}
	}

	T& operator ()(int row, int col) {
		return buf[row * COLS + col];
	}
};

//typedef xf::cv::LineBuffer<KERNEL - 1, WIDTH, uint64_t> linebuf_t;
//typedef xf::cv::Window<KERNEL, KERNEL, uint64_t> window_t;
typedef LineBuffer<KERNEL - 1, WIDTH, uint64_t> linebuf_t;
typedef Window<KERNEL, KERNEL, uint64_t> window_t;

void insert_linebuf(linebuf_t& linebuf, const int x, const uint64_t v) {
	linebuf.shift_pixels_up(x);
	linebuf.insert_bottom_row(v, x);
}

template<int KH, int KW>
void slide_window(linebuf_t& linebuf, window_t& window, const int x, const uint64_t v) {
	uint64_t rows[KH];

	linebuf.get_col(rows, x);
	rows[KH - 1] = v;
	insert_linebuf(linebuf, x, v);

	window.shift_pixels_left();
	window.insert_right_col(rows);
}

template<int KH, int KW>
void pack_window(window_t& window, wpack_t& val) {
	int ptr = 0;
	for (int y = 0; y < KH; y++) {
#pragma HLS unroll
		for (int x = 0; x < KW; x++) {
			uint64_t v = window(y, x);
			val[ptr++] = v;
		}
	}
}

template<int C>
int quick_sum(const cpack_t val) {
	int t[C / 2];
#pragma HLS array_partition variable=t

	int p = 0;
	for (int i = 0; i < C / 2; i++) {
		ap_int<3> v1 = val(p + 2, p);
		p += 4;
		ap_int<3> v2 = val(p + 2, p);
		p += 4;
		t[i] = v1 + v2;
	}
	int d = 1;
	for (int j = 0; j < ilogb(C) - 1; j++) {
		d *= 2;
		for (int i = 0; i < C / 2; i+= d) {
#pragma HLS unroll
			t[i] += t[i + d / 2];
		}
	}
	return t[0];
}

template<int H, int W, int C, int KH, int KW>
void read_input(const int in[H * W * C], wfifo_t& ins) {
	linebuf_t linebuf;
	window_t window;

	int ptr = 0;
	for (int y = 0; y < KH - 1; y++) {
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
			cpack_t val;
			for (int z = 0; z < C; z++) {
#pragma HLS unroll
				uint4_t v = in[ptr++];
				int p = z * 4;
				val(p + 3, p) = v;
			}
			insert_linebuf(linebuf, x, val);
		}
	}
	for (int y = KH - 1; y < H; y++) {
		for (int x = 0; x < KW - 1; x++) {
#pragma HLS pipeline
			cpack_t val;
			for (int z = 0; z < C; z++) {
#pragma HLS unroll
				uint4_t v = in[ptr++];
				int p = z * 4;
				val(p + 3, p) = v;
			}
			slide_window<KH, KW>(linebuf, window, x, val);
		}
		for (int x = KW - 1; x < W; x++) {
#pragma HLS pipeline
			cpack_t val;
			for (int z = 0; z < C; z++) {
#pragma HLS unroll
				uint4_t v = in[ptr++];
				int p = z * 4;
				val(p + 3, p) = v;
			}
			slide_window<KH, KW>(linebuf, window, x, val);
			wpack_t oval;
			pack_window<KH, KW>(window, oval);
			ins.write(oval);
		}
	}
}

template<int F, int KH, int KW, int C, int OH, int OW>
void compute_conv2d_mul(
	const int weight[F * KH * KW * C],
	wfifo_t& ins, wfifo_t& outs)
{
	cpack_t afilter[F * KH * KW];
	cpack_t xfilter[F * KH * KW];
	cpack_t ifilter[F * KH * KW];
#pragma HLS array_partition variable=afilter cyclic factor=KH * KW
#pragma HLS array_partition variable=xfilter cyclic factor=KH * KW
#pragma HLS array_partition variable=ifilter cyclic factor=KH * KW

	int ptr = 0;
	for (int i = 0; i < F * KH * KW; i++) {
#pragma HLS pipeline
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			int v = weight[ptr++];
			int p = z * 4;
			afilter[i](p + 3, p) = (v != 0) ? 0x7 : 0x0;
			xfilter[i](p + 3, p) = (v < 0) ? 0x7 : 0x0;
			ifilter[i](p + 3, p) = (v < 0) ? 0x1 : 0x0;
		}
	}

	for (int i = 0; i < OH * OW; i++) {
		wpack_t ival = ins.read();
		for (int j = 0; j < F; j++) {
#pragma HLS pipeline
			wpack_t oval;
			for (int k = 0; k < KH * KW; k++) {
				oval[k] = ival[k] & afilter[j * KH * KW + k];
				oval[k] ^= xfilter[j * KH * KW + k];
				oval[k] += ifilter[j * KH * KW + k];
			}
			outs.write(oval);
		}
	}
}

template<int M, int F, int KH, int KW, int C, int OH, int OW>
void compute_conv2d_sum(
	const int threshold[M],
	wfifo_t& ins, mfifo_t& outs)
{
	int thr[M];
#pragma HLS array_partition variable=thr

	for (int n = 0; n < M; n++) {
#pragma HLS unroll
		thr[n] = threshold[n];
	}

	for (int i = 0; i < OH * OW; i++) {
		mpack_t oval;
		for (int j = 0; j < F; j++) {
#pragma HLS pipeline
			int acc = 0;
			wpack_t val = ins.read();
			for (int k = 0; k < KH * KW; k++) {
				acc += quick_sum<C>(val[k]);
			}
			uint2_t m = 0;
		        for (int n = 0; n < M; n++) {
#pragma HLS unroll
				if (acc >= thr[n]) {
					m = n + 1;
				}
			}
			int p = j * 2;
			oval(p + 1, p) = m;
		}
		outs.write(oval);
	}
}

template<int C>
void maxpool(mpack_t& val1, const mpack_t val2) {
	for (int z = 0; z < C; z++) {
#pragma HLS unroll
		int p = z * 2;
		uint2_t v1 = val1(p + 1, p);
		uint2_t v2 = val2(p + 1, p);
		if (v2 > v1) {
			val1(p + 1, p) = v2;
		}
	}
}

template<int H, int W, int C>
void compute_maxpool_h(mfifo_t& ins, mfifo_t& outs) {
	for (int xy = 0; xy < H * W; xy += 2) {
#pragma HLS pipeline
		mpack_t val1 = ins.read();
		mpack_t val2 = ins.read();
		maxpool<C>(val1, val2);
		outs.write(val1);
	}
}

template<int H, int W, int C>
void compute_maxpool_v(mfifo_t& ins, mfifo_t& outs) {
	mpack_t buf[W];
#pragma HLS array_partition variable=buf

	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
			mpack_t val = ins.read();
			buf[x] = val;
		}
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
			mpack_t val1 = buf[x];
			mpack_t val2 = ins.read();
			maxpool<C>(val1, val2);
			outs.write(val1);
		}
	}
}

template<int H, int W, int C>
void write_result(int out[H * W * C], mfifo_t& outs) {
	int ptr = 0;
	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS pipeline
		mpack_t val = outs.read();
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			int p = z * 2;
			uint2_t v = val(p + 1, p);
			out[ptr++] = v;
		}
	}
}

void kernel(
	int in[HEIGHT * WIDTH * CHANNEL],
	int weight[FILTER * KERNEL * KERNEL * CHANNEL],
	int threshold[THRESHOLD],
	int out[OHEIGHT * OWIDTH * FILTER])
{
#pragma HLS interface axis port=in
#pragma HLS interface axis port=weight
#pragma HLS interface axis port=threshold
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHANNEL
#pragma HLS array_partition variable=weight cyclic factor=CHANNEL
#pragma HLS array_partition variable=out cyclic factor=FILTER

	wfifo_t ins("input_fifo");
	wfifo_t pips1("pipe_fifo1");
	mfifo_t pips2("pipe_fifo2");
	mfifo_t pips3("pipe_fifo3");
	mfifo_t outs("output_fifo");

#pragma HLS dataflow
	read_input<12, 12, 16, 5, 5>(in, ins);
	compute_conv2d_mul<16, 5, 5, 16, 8, 8>(weight, ins, pips1);
	compute_conv2d_sum<3, 16, 5, 5, 16, 8, 8>(threshold, pips1, pips2);
	compute_maxpool_h<8, 8, 16>(pips2, pips3);
	compute_maxpool_v<4, 4, 16>(pips3, outs);
	write_result<4, 4, 16>(out, outs);
}
