/*
 * 演算回路再利用の検証
 * ・演算回路は最大サイズのConv,Maxpoolを用意し、引数で行列サイズを指定して再利用(ループをbreak?範囲外は0埋め?)
 * ・conv0_wi,conv0_thr -> in -> conv1_wi,conv1_thr -> mat_wi の順にメインメモリからパラメータを転送
 * ・ダブルバッファリングで、パラメータ転送中に演算して演算結果を一時保存
 */
#include "kernel.hpp"
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_streamofblocks.h>
#include <hls_vector.h>

const int WIDTH = 32;
const int HEIGHT = 32;
const int CHANNEL = 16;
const int FILTER = 16;

const int KERNEL = 5;
const int THRESHOLD = 3;

const int FLATTEN = 256;
const int CLASS = 10;
const int CHUNK_SIZE = 16;

template <typename T>
using fifo = hls::stream<T>;

using data_t = int8_t[CHANNEL];
using block_data_t = data_t[HEIGHT * WIDTH];
using win_t = hls::vector<data_t, KERNEL * KERNEL>;
using sob = hls::stream_of_blocks<block_data_t>;

template <int N>
int16_t muladd(const int n, const int8_t vu[N], const int8_t wi[N]) {
	static int16_t t[N];
#pragma HLS array_partition variable=t

	for (int i = 0; i < N; i++) {
		// @see UG1399, Vitis HLS Coding Styles > Loops > Variable Loop Bounds
#pragma HLS unroll
		if (i >= n) break;
		t[i] = vu[i] * wi[i];
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
	using IT = int8_t[K];
	using OT = int16_t[CL];

	void flatten(const IT mat[CL * FL / K], sob& inb, fifo<OT>& pips) {
	    hls::read_lock<block_data_t> inbL(inb);

		int ptr = 0;
		for (int y = 0; y < H; y++) {
			for (int x = 0; x < W; x++) {
				IT vu = inbL[y * WIDTH + x];
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
	void read(fifo<int8_t>& ins, IT mat[CL * FL / K]) {
		int ptr = 0;
		for (int i = 0; i < CL; i++) {
#pragma HLS pipeline
			for (int j = 0; j < FL / K; j++) {
				for (int k = 0; k < K; k++) {
#pragma HLS unroll
					IT val = ins.read();
					mat[j * CL + i][k] = val;
				}
			}
		}
	}

    void compute_and_write_result(int out[CL], const IT mat[CL * FL / K], sob& inb) {
		fifo<OT> pips("pipe_fifo");

#pragma HLS dataflow
		flatten(mat, inb, pips);
		write_result(out, pips);
	}
};

template <int H, int W, int C, typename T>
void read_input(fifo<int8_t>& ins, sob& outb) {
    hls::write_lock<block_data_t> outbL(outb);

	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
			T val;
			for (int z = 0; z < C; z++) {
#pragma HLS unroll
				val[z] = ins.read();
			}
			outbL[y * WIDTH + x] = val;
		}
	}
}

void process(fifo<int8_t>& ins, int out[CLASS]) {
    Dense<CLASS,FLATTEN,CHUNK_SIZE,4,4> matmul0;

	sob even_sob;
	sob odd_sob;

	static data_t mat_wi[CLASS * FLATTEN / CHUNK_SIZE];
#pragma HLS array_partition variable=mat_wi cyclic factor=FLATTEN/CHUNK_SIZE

#pragma HLS dataflow
	matmul0.read(ins, mat_wi);
	read_input<4,4,16,data_t>(ins, even_sob);
	matmul0.compute_and_write_result(out, mat_wi, even_sob);
}

void kernel(int in[256], int matmul0_weight[10 * 256], int out[10]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=16
#pragma HLS array_partition variable=matmul0_weight cyclic factor=16
#pragma HLS array_partition variable=out

    fifo<int8_t> ins;

    for (int i = 0; i < CLASS * FLATTEN; i++) {
#pragma HLS unroll factor=16 skip_exit_check
        ins.write(matmul0_weight[i]);
    }

    for (int i = 0; i < FLATTEN; i++) {
#pragma HLS unroll factor=16 skip_exit_check
        ins.write(in[i]);
    }

    process(ins, out);
}
