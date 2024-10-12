#include "kernel.hpp"

// @see Vitis 高位合成ユーザー ガイド
// https://japan.xilinx.com/support/documentation/sw_manuals_j/xilinx2020_1/ug1399-vitis-hls.pdf
// Vitis HLS 設計手法ガイド > Vitis HLS コーディング スタイル > C++クラスおよび テンプレート > テンプレート > テンプレートを使用した固有のインスタンスの作成 >	再帰関数でのテンプレートの使用
// N = 4, 8 ,16, ...
template<typename T, int N>
struct recursive_t {
	static void merge(T v[SIZE], const int l, const int m, const int r) {
		T t[SIZE];
		int p = 0;
		int i = l;
		int j = m;
		while (i < m && j < r) {
			if (v[i] > v[j]) {
				t[p++] = v[i++];
			} else {
				t[p++] = v[j++];
			}
		}
		if (i == m) {
			while (j < r) {
				t[p++] = v[j++];
			}
		} else {
			while (i < m) {
				t[p++] = v[i++];
			}
		}
		for (int k = 0; k < p; k++) {
			v[l + k] = t[k];
		}
	}

	static void sort(T v[SIZE], const int l, const int r) {
		int m = (l + r) / 2;
		recursive_t<T, N / 2>::sort(v, l, m);
		recursive_t<T, N / 2>::sort(v, m, r);
		recursive_t<T, N>::merge(v, l, m, r);
	}
};

template<typename T>
struct recursive_t<T, 2> {
	static void sort(T v[SIZE], const int l, const int r) {
		T lt = v[l];
		T rt = v[r - 1];
		if (lt < rt) {
			v[l] = rt;
			v[r - 1] = lt;
		}
	}
};

void merge_sort(float buf[SIZE]) {
	recursive_t<float, SIZE>::sort(buf, 0, SIZE);
}

void read_input(const float in[SIZE], float buf[SIZE]) {
	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		buf[i] = in[i];
	}
}

void write_result(float out[SIZE], const float buf[SIZE]) {
	for (int i = 0; i < SIZE; i++) {
#pragma HLS unroll
		out[i] = buf[i];
	}
}

void kernel(const float in[SIZE], float out[SIZE]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in
#pragma HLS array_partition variable=out

	float buf[SIZE];
#pragma HLS array_partition variable=buf

#pragma HLS dataflow
	read_input(in, buf);
	merge_sort(buf);
	write_result(out, buf);
}
