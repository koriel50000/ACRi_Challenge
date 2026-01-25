#pragma once

template <int ROWS, int COLS, typename T, typename WT>
class Window {
private:
	WT buf_;
public:
	void shift_pixels_left() {
#pragma HLS inline
		for (int i = 0; i < ROWS * COLS - 1; i++) {
#pragma HLS unroll
			buf_[i] = buf_[i + 1];
		}
	}

	void insert_right_col(const T value[ROWS]) {
#pragma HLS inline
		for (int i = 0; i < ROWS; i++) {
#pragma HLS unroll
			int idx = (i + 1) * COLS - 1;
			buf_[idx] = value[i];
		}
	}

	WT& get_buf() {
		return buf_;
	}
};

template <int KN, typename T, typename WT>
class LineBuffer256 {
private:
	static const int W = 256;

	T buf_[W * (KN - 1)];
	T last_;
	Window<KN, KN, T, WT> window_;
	int width_;
	int head_;

	void shift_pixels_up_and_insert_bottom_row(T value) {
#pragma HLS inline
		buf_[head_] = value;
		head_++;
		if ((head_ & (W - 1)) >= width_) {
			head_ = (head_ & ~(W - 1)) + W;
			head_ &= (W * (KN - 1) - 1); // KN = 3, 5
		}
		last_ = buf_[head_];
	}

	void get_col(T value[KN - 1]) {
#pragma HLS inline
		value[0] = last_;
		for (int i = 1; i < KN - 1; i++) {
#pragma HLS unroll
			value[i] = buf_[(i * W + head_) & (W * (KN - 1) - 1)];
		}
	}
public:
	void init(int w) {
#pragma HLS array_partition variable=buf_ cyclic=W
		width_ = w;
		head_ = 0;
		last_ = 0;
	}

	void insert_linebuf(const T v) {
		shift_pixels_up_and_insert_bottom_row(v);
	}

	void slide_window(const T v) {
		T rows[KN];
#pragma HLS array_partition variable=rows

		get_col(rows);
		rows[KN - 1] = v;
		shift_pixels_up_and_insert_bottom_row(v);

		window_.shift_pixels_left();
		window_.insert_right_col(rows);
	}

	WT& get_window() {
		return window_.get_buf();
	}
};
