/*
 * 4bit量子化および演算回路再利用の検証
 * ・weightを1bit符号＋3bit指数部の4bitで表現(0,0.125,0.25,0.5,1,2,4,8,NA,-0.125,-0.25,-0.5,-1,-2,-4,-8)
 * ・バッチ正規化後のactivationを1bit符号＋3bit仮数部の4bitで表現(0,1,2,3,4,5,6,7,NA,-1,-2,-3,-4,-5,-6,-7)
 * ・乗算は符号なし3bitの掛け算を、6入力LUTが6個のテーブル参照で計算
 * ・演算回路は最大サイズのConv,Maxpoolを用意し、引数で行列サイズを指定して再利用
 * ・ダブルバッファリングで演算結果を一時保存
 */
#include "kernel.hpp"
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_streamofblocks.h>
#include <hls_vector.h>

#define I64(i) int_t<16>(i)

const int WIDTH = 32;
const int HEIGHT = 32;
const int CHANNEL = 16;
const int FILTER = 16;

const int KERNEL = 5;
const int THRESHOLD = 3;

const int FLATTEN = 256;
const int CLASS = 10;
const int CHUNK_SIZE = 16;

using uint4_t = ap_uint<4>;
using uint6_t = ap_uint<6>;
template <typename T>
using fifo = hls::stream<T>;

template <int N, int W = 4>
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

	inline ap_range_ref<W*N, false> operator[](size_t index) const {
		assert(index < N);
		return buf_(W * index + W - 1, W * index);
	}

	inline ap_range_ref<W*N, false> operator[](size_t index) {
		assert(index < N);
		return buf_(W * index + W - 1, W * index);
	}
};

uint6_t mul66(const uint6_t i) {
	static const uint6_t table[] = {
		0,	0,	0,	0,	0,	0,	0,	0,
		0,	0,	0,	1,	1,	2,	4,	8,
		0,	0,	1,	1,	2,	4,	8,	16,
		0,	0,	1,	2,	3,	6,	12,	24,
		0,	1,	1,	2,	4,	8,	16,	32,
		0,	1,	1,	3,	5,	10,	20,	40,
		0,	1,	2,	3,	6,	12,	24,	48,
		0,	1,	2,	4,	7,	14,	28,	56,
	};
	return table[i];
}

int8_t mul(const uint4_t v, const uint4_t w) {
	uint6_t oval = mul66((v(2, 0), w(2, 0)));
	return (v[3] ^ w[3]) == 1 ? (-oval).to_int() : oval.to_int();
}

template <int N>
int16_t muladd(const int n, const int_t<N> vu, const int_t<N> wi) {
	static int16_t t[N];
#pragma HLS array_partition variable=t

	for (int i = 0; i < N; i++) {
		// @see UG1399, Vitis HLS Coding Styles > Loops > Variable Loop Bounds
#pragma HLS unroll
		if (i >= n) break;
		t[i] = mul(vu[i], wi[i]);
	}

	for (int d = 1; d < N; d *= 2) {
		if (d >= n) break;
		for (int i = 0; i < N; i += d * 2) {
#pragma HLS unroll
			if (i >= n) break;
			t[i] += t[i + d];
		}
	}
	return t[0];
}

template <int CL, int FL, int K, int H, int W>
class Dense {
private:
	using IT = int_t<K>;
	using OT = int_t<CL,16>;

	void flatten(const IT mat[CL * FL / K], const IT inb[], fifo<OT>& pips) {
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

	void write_result(int out[CL], fifo<OT>& pips) {
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
	
		for (int i = 0; i < CL; i++) {
#pragma HLS unroll
			out[i] = acc[i];
		}
	}
public:
	void compute_and_write_result(int out[CL], const IT mat[CL * FL / K], const IT inb[]) {
		fifo<OT> pips("pipe_fifo");

#pragma HLS dataflow
		flatten(mat, inb, pips);
		write_result(out, pips);
	}
};

template <int H, int W, int C, typename T>
void read_input(const int in[H * W * C], T inb[]) {
	int ptr = 0;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
			T val;
			for (int z = 0; z < C; z++) {
#pragma HLS unroll
				val[z] = in[ptr++];
			}
			inb[y * WIDTH + x] = val;
		}
	}
}

using data_t = int_t<CHANNEL>;
using block_data_t = data_t[HEIGHT * WIDTH];
using win_t = hls::vector<data_t, KERNEL * KERNEL>;
template <typename T>
using sob = hls::stream_of_blocks<T>;

Dense<CLASS,FLATTEN,CHUNK_SIZE,4,4> matmul0;

void kernel(int in[256], int matmul0_weight[10 * 256], int out[10]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=16
#pragma HLS array_partition variable=out

	static block_data_t even_buf;
	static block_data_t odd_buf;
#pragma HLS array_partition variable=even_buf cyclic factor=WIDTH
#pragma HLS array_partition variable=odd_buf cyclic factor=WIDTH

	static data_t mat_wi[CLASS * FLATTEN / CHUNK_SIZE] = {
I64(0x0c00000000000000), I64(0x0000c00cccc00000), I64(0xc0cc0040c4c004c0), I64(0xc0ccc400c4c0c4cc),
I64(0x4c04000000400c00), I64(0x000000004c400000), I64(0x0c000c0c00000c00), I64(0x000000c400000000),
I64(0x00040cc000400000), I64(0x000000c44c040004), I64(0xc00000c00c000000), I64(0x00000c004c004000),
I64(0x04c0c0000000c0cc), I64(0x040c00c0c0c00000), I64(0x0c00000000000c00), I64(0x0000000400000000),
I64(0x0c000c0000000c00), I64(0x400c04c000ccc400), I64(0x00000c0004000000), I64(0x000000c40c040004),
I64(0x00000c00cc00c000), I64(0x0c40004444000c00), I64(0x0400c00c00000000), I64(0x0000000000004004),
I64(0x0c040c004c040c00), I64(0xc00c040000c0c400), I64(0xcc00044c44000000), I64(0x40004400000c0440),
I64(0x040000c0c0000000), I64(0x04000cc0c00c0000), I64(0x44000cc00000c000), I64(0xcc00cc0000040000),
I64(0x0000c00000000000), I64(0x00000c00c0000000), I64(0xcc040c0044044c04), I64(0x04cc0400c0ccc00c),
I64(0xc00c040c000c0c0c), I64(0x0c04440000000440), I64(0x0400000000004000), I64(0x400c4000000c0400),
I64(0x0c040c000c400044), I64(0x000c0c000cccc0cc), I64(0x000c040004c00000), I64(0x00ccc400c0c0000c),
I64(0x00040cc00c400004), I64(0x00000cc40c000000), I64(0x0c440c004c400c00), I64(0xc0ccc44004c00000),
I64(0x000000c0000000c0), I64(0x40040c0000400000), I64(0xc0c00c4c00000c00), I64(0x00044ccc4c000000),
I64(0x0000040000c00000), I64(0x000004c00c044400), I64(0x40044c00004c0400), I64(0x0000004000000000),
I64(0x0c004c0400400000), I64(0xc0c0044c00000000), I64(0x4000c0c000000000), I64(0x00000004c000000c),
I64(0xc000c00cc0000ccc), I64(0x0c400c4444400c00), I64(0xc004000040000c00), I64(0x00040c000c00c004),
I64(0x00004cc440000000), I64(0x00cc0440c0ccc00c), I64(0x000c400400000000), I64(0xc40000cc00000cc0),
I64(0x000c00c0c0000400), I64(0x000000c0c0000000), I64(0x0404c0cc0c0440c4), I64(0x000ccc0cc0c000cc),
I64(0x0c00000040040c00), I64(0x00000000c00000c0), I64(0x0c00000404040004), I64(0x00cc040000ccc00c),
I64(0x400044c40c0c4440), I64(0x04000c0004000000), I64(0xc0000000c0000000), I64(0x0040cc000c00c000),
I64(0x4c440c0000400c04), I64(0x00000c0000c00000), I64(0x0044400000000044), I64(0x000c000000c0c00c),
I64(0x00c0cc00c0000400), I64(0x000c000004c0c000), I64(0x00040cc040400000), I64(0x00cc0000c00000c0),
I64(0xc00400cc0c004c00), I64(0x00cccc00000000cc), I64(0x4c000040040c00c0), I64(0xcc040c0000440000),
I64(0x000000c00c000044), I64(0x000cc00c00c4c00c), I64(0x00cc0000cc0c0000), I64(0x000c040004000000),
I64(0x40440c0000000000), I64(0x0000404c40000000), I64(0x00440c0c0c000c04), I64(0x00cc00c0c000000c),
I64(0xc00c40c040c00000), I64(0x0c00cc040404cc00), I64(0x4000000000000400), I64(0x0000040004000000),
I64(0x000c0c0000000000), I64(0x0000040000000000), I64(0x000c00c000004000), I64(0xcc04000000040c40),
I64(0x000ccc4cc40cc0cc), I64(0x00040c0c00000000), I64(0xcc0000c00c040004), I64(0x0000000cc0c0400c),
I64(0x000c400440000000), I64(0x0000000000000000), I64(0x04c00c00c000c000), I64(0x0000000000000000),
I64(0x004000000c004404), I64(0x00000c000000c0c0), I64(0x000c040000ccc0cc), I64(0xc000cc4c0000ccc0),
I64(0x0040cc0000000004), I64(0xc0004c0c00000000), I64(0x00000c0000000000), I64(0x40000404040c0440),
I64(0xc0c0004cc0000ccc), I64(0x4000040404000400), I64(0xc0c00cc0cc0000c0), I64(0xc000000c0c000c00),
I64(0x00440cc000400000), I64(0x00cc004c00c00c00), I64(0x00004004c00c0400), I64(0x0004000000440c00),
I64(0x000cc0000000000c), I64(0x000c04c000c00040), I64(0xc000040c0c004c00), I64(0x0000000000000040),
I64(0x40ccccc0c0c0c0c0), I64(0xcc444c0c00404c00), I64(0x4040cc0000000400), I64(0xc000040c40000c00),
I64(0x000c000000c00440), I64(0x00000c0404000000), I64(0x04cc000000ccc00c), I64(0x000404000c000040),
I64(0x0000c00c00400cc0), I64(0x004040000c000000), I64(0x00cc0cc0cc0cc40c), I64(0xcc000c0004000c00),
I64(0x0040c00000000400), I64(0x0c00000c40040c00), I64(0xc0c000000004c000), I64(0x0000000000c0000c),
I64(0x400c44c4000cc40c), I64(0x0000000000000004), I64(0x0000000004000cc0), I64(0x000000000c004000),
I64(0x00cc0c0c0000c400), I64(0xc0c0c40004c00cc0), I64(0x0040000000400004), I64(0x0c040c0000004c00),
	};
#pragma HLS array_partition variable=mat_wi cyclic factor=CLASS

	read_input<4,4,16,data_t>(in, even_buf);
	matmul0.compute_and_write_result(out, mat_wi, even_buf);
}
