#pragma once
#include <cassert>
#include <cstdint>

template <int ROWS, int COLS, typename T>
class Window {
private:
	T buf[ROWS * COLS];
public:
	void reset_center_col() {
#pragma HLS array_partition variable=buf complete
		reset_center_col: for (int i = 0; i < ROWS; i++) {
#pragma HLS unroll
			int idx = i * COLS + 1;
			buf[idx] = 0;
		}
	}

	void shift_pixels_left() {
		shift_pixels_left: for (int i = 0; i < ROWS * COLS - 1; i++) {
#pragma HLS unroll
			buf[i] = buf[i + 1];
		}
	}

	void insert_right_col(const int h, const int w, const bool depthwise, const T *inb, const int y0, const int x0) {
		int x = x0;
		int y = y0 - 1;
		insert_right_col: for (int i = 0; i < ROWS; i++) {
#pragma HLS unroll
			int idx = (i + 1) * COLS - 1;
			if (0 <= x && x < w	&& 0 <= y && y < h) {
				int ptr = y * w + x;
				if (depthwise) {
					buf[idx] = inb[ptr];
				} else {
					T val = inb[ptr >> 2];
					buf[idx] = T((uint64_t)val.word(ptr & 0x3));
				}
			} else {
				buf[idx] = 0;
			}
			y++;
		}
	}

    inline const T& operator[](size_t index) const {
		assert(index < ROWS * COLS);
        return buf[index];
    }
};

//// W = 8, 16, 32, 64, 128, 256, ...
//template <int W, int KN, typename T, typename WT>
//class LineBuffer {
//private:
//	T buf_[W * (KN - 1)];
//	Window<KN, KN, T, WT> window_;
//	int width_;
//	int head_;
//
//	void shift_pixels_up_and_insert_bottom_row(T value) {
//#pragma HLS inline
//		buf_[head_] = value;
//		head_++;
//	    if ((head_ & (W - 1)) >= width_) {
//			head_ = (head_ & ~(W - 1)) + W;
//			head_ &= (W * (KN - 1) - 1); // KN = 3, 5
//		}
//	}
//
//	void get_col(T value[KN - 1]) {
//#pragma HLS inline
//		get_col: for (int i = 0; i < KN - 1; i++) {
//#pragma HLS unroll
//			value[i] = buf_[(head_ + i * W) & (W * (KN - 1) - 1)];
//		}
//	}
//public:
//	LineBuffer() : buf_{} {}
//
//	void reset(int w) {
//#pragma HLS bind_storage variable=buf_ type=ram_t2p
//		width_ = w;
//		head_ = 0;
//	}
//
//	void insert_linebuf(const T v) {
//		shift_pixels_up_and_insert_bottom_row(v);
//	}
//
//	void slide_window(const T v) {
//		T rows[KN];
//#pragma HLS array_partition variable=rows complete
//
//		get_col(rows);
//		rows[KN - 1] = v;
//		shift_pixels_up_and_insert_bottom_row(v);
//
//		window_.shift_pixels_left();
//		window_.insert_right_col(rows);
//	}
//
//	WT& get_window() {
//		return window_.get_buf();
//	}
//};
