#include "kernel.hpp"
#include <hls_math.h>

const int TOP5 = 5;
const int DEGREE = 8;

typedef float top5_t[TOP5];

void shift0(top5_t v, const float f) {
	v[4] = v[3];
	v[3] = v[2];
	v[2] = v[1];
	v[1] = v[0];
	v[0] = f;
}

void shift1(top5_t v, const float f) {
	v[4] = v[3];
	v[3] = v[2];
	v[2] = v[1];
	v[1] = f;
}

void shift2(top5_t v, const float f) {
	v[4] = v[3];
	v[3] = v[2];
	v[2] = f;
}

void shift3(top5_t v, const float f) {
	v[4] = v[3];
	v[3] = f;
}

void shift4(top5_t v, const float f) {
	v[4] = f;
}

void insert_sort(top5_t v, const float f) {
	bool b0 = (f > v[0]);
	bool b1 = (f > v[1]);
	bool b2 = (f > v[2]);
	bool b3 = (f > v[3]);
	bool b4 = (f > v[4]);
	if (b0) {
		shift0(v, f);
	} else if (b1) {
		shift1(v, f);
	} else if (b2) {
		shift2(v, f);
	} else if (b3) {
		shift3(v, f);
	} else if (b4) {
		shift4(v, f);
	}
}

void shift0(int ptr[TOP5], const int p) {
	ptr[4] = ptr[3];
	ptr[3] = ptr[2];
	ptr[2] = ptr[1];
	ptr[1] = ptr[0];
	ptr[0] = p;
}

void shift1(int ptr[TOP5], const int p) {
	ptr[4] = ptr[3];
	ptr[3] = ptr[2];
	ptr[2] = ptr[1];
	ptr[1] = p;
}

void shift2(int ptr[TOP5], const int p) {
	ptr[4] = ptr[3];
	ptr[3] = ptr[2];
	ptr[2] = p;
}

void shift3(int ptr[TOP5], const int p) {
	ptr[4] = ptr[3];
	ptr[3] = p;
}

void shift4(int ptr[TOP5], const int p) {
	ptr[4] = p;
}

void insert_ptr_sort(const float buf[TOP5 * DEGREE], int ptr[DEGREE], int p) {
	float f = buf[p];
	bool b0 = (f > buf[ptr[0]]);
	bool b1 = (f > buf[ptr[1]]);
	bool b2 = (f > buf[ptr[2]]);
	bool b3 = (f > buf[ptr[3]]);
	bool b4 = (f > buf[ptr[4]]);
	if (b0) {
		shift0(ptr, p);
	} else if (b1) {
		shift2(ptr, p);
	} else if (b2) {
		shift2(ptr, p);
	} else if (b3) {
		shift3(ptr, p);
	} else if (b4) {
		shift4(ptr, p);
	}
}

int lshift(int ptr[TOP5]) {
	int p = ptr[0];
	ptr[0] = ptr[1];
	ptr[1] = ptr[2];
	ptr[2] = ptr[3];
	ptr[3] = ptr[4];
	return p;
}

// @see Vitis 高位合成ユーザー ガイド
// https://japan.xilinx.com/support/documentation/sw_manuals_j/xilinx2020_1/ug1399-vitis-hls.pdf
// Vitis HLS 設計手法ガイド > Vitis HLS コーディング スタイル > C++クラスおよび テンプレート > テンプレート > テンプレートを使用した固有のインスタンスの作成 >	再帰関数でのテンプレートの使用
// N = 4, 8 ,16, ...
template<int N>
struct recursive_t {
	static void merge(const float buf[TOP5 * DEGREE], int ptr[DEGREE],
			const int l, const int m, const int r) {
		int t[DEGREE];
		int p = 0;
		int i = l;
		int j = m;
		while (i < m && j < r) {
			if (buf[ptr[i]] > buf[ptr[j]]) {
				t[p++] = ptr[i++];
			} else {
				t[p++] = ptr[j++];
			}
		}
		if (i == m) {
			while (j < r) {
				t[p++] = ptr[j++];
			}
		} else {
			while (i < m) {
				t[p++] = ptr[i++];
			}
		}
		for (int k = 0; k < p; k++) {
			ptr[l + k] = t[k];
		}
	}

	static void sort(const float buf[TOP5 * DEGREE], int ptr[DEGREE],
			const int l, const int r) {
		int m = (l + r) / 2;
		recursive_t<N / 2>::sort(buf, ptr, l, m);
		recursive_t<N / 2>::sort(buf, ptr, m, r);
		recursive_t<N>::merge(buf, ptr, l, m, r);
	}
};

template<>
struct recursive_t<2> {
	static void sort(const float buf[TOP5 * DEGREE], int ptr[DEGREE],
			const int l, const int r) {
		int lt = ptr[l];
		int rt = ptr[r - 1];
		if (buf[lt] < buf[rt]) {
			ptr[l] = rt;
			ptr[r - 1] = lt;
		}
	}
};

void merge_ptr_sort(const float buf[TOP5 * DEGREE], int ptr[DEGREE]) {
	recursive_t<DEGREE>::sort(buf, ptr, 0, DEGREE);
}

int quick_max(const float buf[TOP5 * DEGREE], const int ptr[DEGREE]) {
	int p[DEGREE / 2];
	int len = 2;
	for (int i = 0; i < DEGREE; i += len) {
		int hi = i;
		int lo = i + 1;
		p[i / 2] = (buf[ptr[hi]] >= buf[ptr[lo]]) ? hi : lo;
	}

	for (int i = 0; i < DEGREE / 2; i += len) {
		int hi = p[i];
		int lo = p[i + len / 2];
		p[i] = (buf[ptr[hi]] >= buf[ptr[lo]]) ? hi : lo;
	}

	len *= 2;
	for (int i = 0; i < DEGREE / 2; i += len) {
		int hi = p[i];
		int lo = p[i + len / 2];
		p[i] = (buf[ptr[hi]] >= buf[ptr[lo]]) ? hi : lo;
	}

	return p[0];
}

//void debug(const float buf[TOP5 * DEGREE], const int ptr[DEGREE]) {
//	for (int d = 0; d < DEGREE; d++) {
//		printf("%d: ", d);
//		for (int i = 0; i < TOP5; i++) {
//			printf("%.4f ", buf[TOP5 * d + i]);
//		}
//		printf("\n");
//	}
//	for (int i = 0; i < TOP5; i++) {
//		printf("%.4f ", buf[ptr[i]]);
//	}
//	printf("\n");
//}

void kernel(const float in[SIZE], float out[5]) {
#pragma HLS interface axis port=in
#pragma HLS interface axis port=out
#pragma HLS array_partition variable=in cyclic factor=DEGREE
#pragma HLS array_partition variable=out

	static float buf[TOP5 * DEGREE];
	static int ptr[DEGREE];
#pragma HLS array_partition variable=buf
#pragma HLS array_partition variable=ptr

	for (int d = 0; d < DEGREE; d++) {
#pragma HLS unroll
		for (int i = 0; i < TOP5; i++) {
			buf[TOP5 * d + i] = -2000.0;
		}
		ptr[d] = TOP5 * d;
	}

	for (int i = 0; i < SIZE; i += DEGREE) {
		float chunk[DEGREE];
		for (int d = 0; d < DEGREE; d++) {
			chunk[d] = in[i + d];
		}
		for (int d = 0; d < DEGREE; d++) {
			insert_sort(&buf[TOP5 * d], chunk[d]);
		}
	}

	merge_ptr_sort(buf, ptr);
	//debug(buf, ptr);

	for (int i = 0; i < TOP5; i++) {
		out[i] = buf[ptr[0]++];
		//printf("ptr=%d out[%d]=%.4f\n", ptr[0], i, out[i]);
		int next_ptr = lshift(ptr);
		insert_ptr_sort(buf, ptr, next_ptr);
		//debug(buf, ptr);
		//printf("\n");
	}
}
