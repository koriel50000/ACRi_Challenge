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
const int CHUNK_SIZE = 16; // == CHANNEL

using data_t = hls::vector<int8_t, CHANNEL>;
using block_data_t = data_t[HEIGHT * WIDTH];
using block_mat_t = data_t[CLASS * FLATTEN / CHUNK_SIZE];
using win_t = hls::vector<data_t, KERNEL * KERNEL>;

template <typename T>
using fifo = hls::stream<T>;
template <typename T>
using sob = hls::stream_of_blocks<T>;

int16_t muladd(const data_t& vu, const data_t& wi) {
	static int16_t t[CHANNEL];
#pragma HLS array_partition variable=t

	for (int i = 0; i < CHANNEL; i++) {
		// @see UG1399, Vitis HLS Coding Styles > Loops > Variable Loop Bounds
#pragma HLS unroll
		t[i] = vu[i] * wi[i];
	}

	for (int d = 1; d < CHANNEL; d *= 2) {
		for (int i = 0; i < CHANNEL; i += d * 2) {
#pragma HLS unroll
			t[i] += t[i + d];
		}
	}
	return t[0];
}

template <int CL, int FL, int K, int H, int W>
class Dense {
private:
	using IT = hls::vector<int8_t, K>;
	using OT = hls::vector<int16_t, CL>;

	void flatten(block_mat_t& mat_wi, block_data_t& in_buf, fifo<OT>& pips) {
		int ptr = 0;
		for (int y = 0; y < H; y++) {
			for (int x = 0; x < W; x++) {
				IT& vu = in_buf[y * WIDTH + x];
				OT oval;
				for (int i = 0; i < CL; i++) {
#pragma HLS pipeline
					IT& wi = mat_wi[ptr++];
					int16_t acc = muladd(vu, wi);
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
	void compute_and_write_result(int out[CL], block_mat_t& mat_wi, block_data_t& in_buf) {
		fifo<OT> pips("pipe_fifo");

#pragma HLS dataflow
		flatten(mat_wi, in_buf, pips);
		write_result(out, pips);
	}
};

template <int H, int W, int C, int CL, int FL, int K>
void read_input(const int in[], const int matmul0_weight[],
    block_mat_t& mat_wi, block_data_t& out_buf, fifo<bool>& ends)
{
    int ptr = 0;
	for (int i = 0; i < CL; i++) {
#pragma HLS pipeline
		for (int j = 0; j < FL / K; j++) {
			for (int k = 0; k < K; k++) {
#pragma HLS unroll
				mat_wi[j * CL + i][k] = matmul0_weight[ptr++];
			}
		}
	}

    ptr = 0;
	for (int y = 0; y < H; y++) {
		for (int x = 0; x < W; x++) {
#pragma HLS pipeline
			for (int z = 0; z < C; z++) {
#pragma HLS unroll
				out_buf[y * WIDTH + x][z] = in[ptr++];
			}
		}
	}

	ends.write(true);
}

void kernel(int in[256], int matmul0_weight[10 * 256], int out[10]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=16
#pragma HLS array_partition variable=matmul0_weight cyclic factor=16
#pragma HLS array_partition variable=out

    fifo<bool> ends;
	static block_data_t even_buf;
	static block_data_t odd_buf;
	static block_mat_t mat_wi;
#pragma HLS array_partition variable=even_buf cyclic factor=WIDTH
#pragma HLS array_partition variable=odd_buf cyclic factor=WIDTH
#pragma HLS array_partition variable=mat_wi cyclic factor=FLATTEN/CHUNK_SIZE

    Dense<CLASS,FLATTEN,CHUNK_SIZE,4,4> matmul0;

#pragma HLS dataflow
	read_input<4,4,16,10,256,16>(in, matmul0_weight, mat_wi, even_buf, ends);
	ends.read();
	matmul0.compute_and_write_result(out, mat_wi, even_buf);
}
