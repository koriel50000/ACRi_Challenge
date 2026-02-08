#pragma once

template <int ROWS, int COLS, typename T, typename WT>
class Window {
private:
	WT buf_;
public:
	Window() : buf_{} {}

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

// W = 8, 16, 32, 64, 128, 256, ...
template <int W, int KN, typename T, typename WT>
class LineBuffer {
private:
	T buf_[W * (KN - 1)];
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
	}

	void get_col(T value[KN - 1]) {
#pragma HLS inline
		for (int i = 0; i < KN - 1; i++) {
#pragma HLS unroll
			value[i] = buf_[(head_ + i * W) & (W * (KN - 1) - 1)];
		}
	}
public:
	LineBuffer() : buf_{} {}

	void reset(int w) {
#pragma HLS bind_storage variable=buf_ type=ram_2p impl=bram
		width_ = w;
		head_ = 0;
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
