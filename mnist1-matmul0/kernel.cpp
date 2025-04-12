/*
 * 4bit量子化および演算回路再利用の検証
 * ・weightを1bit符号＋3bit指数部の4bitで表現(0,0.125,0.25,0.5,1,2,4,8,NA,-0.125,-0.25,-0.5,-1,-2,-4,-8)
 * ・バッチ正規化後のactivationを1bit符号＋3bit仮数部の4bitで表現(0,1,2,3,4,5,6,7,NA,-1,-2,-3,-4,-5,-6,-7)
 * ・乗算は符号なし3bitの掛け算を、6入力LUTが6個のテーブル参照で計算
 * ・演算回路は最大サイズのConv,Maxpoolを用意し、引数で行列サイズを指定して再利用
 * ・ダブルバッファリングで演算結果を一時保存
 */
#include "kernel.hpp"
#include <stdint.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

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
		// @see UG1399 Vitis HLS Coding Styles > Loops > Variable Loop Bounds
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
				IT vu = inb[y * W + x];
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
	void read(const int weight[CL * FL], IT mat[CL * FL / K]) {
		int ptr = 0;
		for (int i = 0; i < CL; i++) {
#pragma HLS pipeline
			for (int j = 0; j < FL / K; j++) {
				for (int k = 0; k < K; k++) {
#pragma HLS unroll
					uint4_t val = (weight[ptr++] << 2) & 0xf;
					mat[j * CL + i][k] = val;
				}
			}
		}
	}

	void compute_and_write_result(int out[CL], const IT mat[CL * FL / K], const IT inb[]) {
		fifo<OT> pips("pipe_fifo");

#pragma HLS dataflow
		flatten(mat, inb, pips);
		write_result(out, pips);
	}
};

template <int FL, int K>
void read_input(const int in[FL], int_t<K> inb[FL / K]) {
	int ptr = 0;
	for (int j = 0; j < FL / K; j++) {
#pragma HLS pipeline
		int_t<K> val;
		for (int k = 0; k < K; k++) {
#pragma HLS unroll
			val[k] = in[ptr++];
		}
		inb[j] = val;
	}
}

void kernel(int in[256], int weight[10 * 256], int out[10]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=16
#pragma HLS array_partition variable=weight cyclic factor=16
#pragma HLS array_partition variable=out

	static int_t<CHUNK_SIZE> buf[FLATTEN / CHUNK_SIZE];
#pragma HLS array_partition variable=buf cyclic factor=FLATTEN / CHUNK_SIZE

	static int_t<CHUNK_SIZE> mat_wi[CLASS * FLATTEN / CHUNK_SIZE];
#pragma HLS array_partition variable=mat_wi

	Dense<CLASS,FLATTEN,CHUNK_SIZE,4,4> matmul0;

	read_input<FLATTEN,CHUNK_SIZE>(in, buf);
	matmul0.read(weight, mat_wi);
	matmul0.compute_and_write_result(out, mat_wi, buf);
}
