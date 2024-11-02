#include "kernel.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

const int WIDTH = 24;
const int HEIGHT = 24;
const int CHANNEL = 16;

const int OWIDTH = WIDTH / 2;
const int OHEIGHT = HEIGHT / 2;

using uint2_t = ap_uint<2>;

template <int W, int N>
class int_t {
private:
	ap_uint<W*N> buf_;
public:
	int_t() : buf_(0) {}
	int_t(int i) : buf_(i) {}
	int_t(unsigned int ui) : buf_(ui) {}
	int_t(long l) : buf_(l) {}
	int_t(unsigned long ul) : buf_(ul) {}
	int_t(const char* s) : buf_(s) {}

	ap_range_ref<W*N, false> operator[](size_t index) const {
		assert(index < N);
		return buf_(W * index + W - 1, W * index);
	}

	ap_range_ref<W*N, false> operator[](size_t index) {
		assert(index < N);
		return buf_(W * index + W - 1, W * index);
	}
};

template <typename T>
using fifo = hls::stream<T>;

template <typename T, int H, int W, int C>
class MaxPool2x2 {
private:
	void maxpool(const T v1, const T v2, T& ov) {
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			ov[z] = (v1[z] > v2[z]) ? v1[z] : v2[z];
		}
	}
public:
	void compute_h(fifo<T>& ins, fifo<T>& outs) {
		for (int xy = 0; xy < H * W / 2; xy++) {
#pragma HLS pipeline
			T val1 = ins.read();
			T val2 = ins.read();
			T oval;
			maxpool(val1, val2, oval);
			outs.write(oval);
		}
	}

	void compute_v(fifo<T>& ins, fifo<T>& outs) {
		T buf[W / 2];
#pragma HLS array_partition variable=buf

		for (int y = 0; y < H / 2; y++) {
#pragma HLS pipeline
			for (int x = 0; x < W / 2; x++) {
				buf[x] = ins.read();
			}
			for (int x = 0; x < W / 2; x++) {
				T val1 = buf[x];
				T val2 = ins.read();
				T oval;
				maxpool(val1, val2, oval);
				outs.write(oval);
			}
		}
	}
};

using MaxPool0 = MaxPool2x2<int_t<2,16>, 24, 24, 16>;

template<int H, int W, int C>
void read_input(const int in[H * W * C], fifo<int_t<2,16>>& ins) {
	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS pipeline
		int_t<2,16> val;
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			val[z] = in[xy * C + z];
		}
		ins.write(val);
	}
}

template<int H, int W, int C>
void write_result(int out[H * W * C], fifo<int_t<2,16>>& outs) {
	for (int xy = 0; xy < H * W; xy++) {
#pragma HLS pipeline
		int_t<2,16> val = outs.read();
		for (int z = 0; z < C; z++) {
#pragma HLS unroll
			out[xy * C + z] = val[z];
		}
	}
}

void kernel(int in[HEIGHT * WIDTH * CHANNEL],
	int out[OHEIGHT * OWIDTH * CHANNEL])
{
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHANNEL
#pragma HLS array_partition variable=out cyclic factor=CHANNEL

	fifo<int_t<2,16>> ins("input_fifo");
	fifo<int_t<2,16>> pips("pipe_fifo");
	fifo<int_t<2,16>> outs("output_fifo");

	MaxPool0 maxpool0;

#pragma HLS dataflow
	read_input<24, 24, 16>(in, ins);
	maxpool0.compute_h(ins, pips);
	maxpool0.compute_v(pips, outs);
	write_result<12, 12, 16>(out, outs);
}
