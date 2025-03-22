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

const int FLATTEN = 256;
const int CLASS = 10;

const int CHUNK_SIZE = 16;

using uint4_t = ap_uint<4>;
using uint6_t = ap_uint<6>;

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

	inline ap_range_ref<W*N, false> operator[](size_t index) const {
		assert(index < N);
		return buf_(W * index + W - 1, W * index);
	}

	inline ap_range_ref<W*N, false> operator[](size_t index) {
		assert(index < N);
		return buf_(W * index + W - 1, W * index);
	}
};

void mul66(uint6_t i, uint6_t& o) {
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
	o = table[i];
}

int8_t mul(uint4_t v, uint4_t w) {
	uint6_t oval;
	mul66((v(2, 0), w(2, 0)), oval);
	return (v[3] ^ w[3] == 1) ? (-oval).to_int() : oval.to_int();
}

int16_t muladd16(int_t<4,16> vu, int_t<4,16> wi) {
	int16_t t[16];
#pragma HLS array_partition variable=t

	for (int i = 0; i < 16; i++) {
#pragma HLS unroll
		t[i] = mul(vu[i], wi[i]);
	}

	for (int d = 1; d < 16; d *= 2) {
		for (int i = 0; i < 16; i += d * 2) {
#pragma HLS unroll
			t[i] += t[i + d];
		}
	}
	return t[0];
}

template <int CL, int FL, int K>
class Dense {
private:
	int_t<4,K> mat[CL * FL / K];
public:
	void read(const int weight[CL * FL]) {
#pragma HLS array_partition variable=mat
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

	void compute(int_t<4,K> inb[FL / K], int_t<16,CL> outb[FL / K]) {
		int ptr = 0;
		for (int j = 0; j < FL / K; j++) {
#pragma HLS pipeline
			int_t<4,K> vu = inb[j];
			for (int i = 0; i < CL; i++) {
				int_t<4,K> wi = mat[ptr++];
				int16_t acc = muladd16(vu, wi);
				outb[j][i] = acc;
			}
		}
	}
};

template <int FL, int K>
void read_input(const int in[FL], int_t<4,K> inb[FL / K]) {
	int ptr = 0;
	for (int j = 0; j < FL / K; j++) {
#pragma HLS pipeline
		int_t<4,K> val;
		for (int k = 0; k < K; k++) {
#pragma HLS unroll
			val[k] = in[ptr++];
		}
		inb[j] = val;
	}
}

template <int CL, int FL, int K>
void write_result(int out[CL], int_t<16,CL> outb[FL / K]) {
	static int16_t acc[CL];
#pragma HLS array_partition variable=acc

	for (int i = 0; i < CL; i++) {
#pragma HLS unroll
		acc[i] = 0;
	}

	for (int j = 0; j < FL / K; j++) {
#pragma HLS pipeline
		int_t<16,CL> val = outb[j];
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

void kernel(int in[FLATTEN], int weight[CLASS * FLATTEN], int out[CLASS]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=weight cyclic factor=CHUNK_SIZE
#pragma HLS array_partition variable=out

	static int_t<4,CHUNK_SIZE> even_buf[FLATTEN / CHUNK_SIZE];
	static int_t<16,CLASS> odd_buf[FLATTEN / CHUNK_SIZE];
#pragma HLS array_partition variable=even_buf
#pragma HLS array_partition variable=odd_buf

	Dense<CLASS,FLATTEN,CHUNK_SIZE> matmul0;

#pragma HLS pipeline

	read_input<FLATTEN,CHUNK_SIZE>(in, even_buf);
	matmul0.read(weight);
	matmul0.compute(even_buf, odd_buf);
	write_result<CLASS,FLATTEN,CHUNK_SIZE>(out, odd_buf);
}
