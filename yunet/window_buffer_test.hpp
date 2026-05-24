#include <iostream>
#include <hls_vector.h>
#include "window_buffer.hpp"
#include "xunit.hpp"

using namespace Assertions;

class WindowBufferTest
{
public:
	static void test_window()
	{
//		int inb[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
//		Window<3, 3, int> window;
//		const int h = 3;
//		const int w = 3;
//
//		window.reset_center_col();
//		window.insert_right_col(h, w, true, inb, 0, 0);
//		window.shift_pixels_left();
//		window.insert_right_col(h, w, true, inb, 0, 1);
////		assertArrayEquals({0, 0, 0, 0, 1, 2, 0, 4, 5}, window.get_buf());
//
//		window.shift_pixels_left();
//		window.insert_right_col(h, w, true, inb, 0, 2);
////		assertArrayEquals({0, 0, 0, 1, 2, 3, 4, 5, 6}, window.get_buf());
//
//		window.shift_pixels_left();
//		window.insert_right_col(h, w, true, inb, 0, 3);
////		assertArrayEquals({0, 0, 0, 2, 3, 0, 5, 6, 0}, window.get_buf());
//
//		window.reset_center_col();
//		window.insert_right_col(h, w, true, inb, 1, 0);
//		window.shift_pixels_left();
//		window.insert_right_col(h, w, true, inb, 1, 1);
////		assertArrayEquals({0, 1, 2, 0, 4, 5, 0, 7, 8}, window.get_buf());
//
//		window.shift_pixels_left();
//		window.insert_right_col(h, w, true, inb, 1, 2);
////		assertArrayEquals({1, 2, 3, 4, 5, 6, 7, 8, 9}, window.get_buf());
//
//		window.shift_pixels_left();
//		window.insert_right_col(h, w, true, inb, 1, 3);
////		assertArrayEquals({2, 3, 0, 5, 6, 0, 8, 9, 0}, window.get_buf());
//
//		window.reset_center_col();
//		window.insert_right_col(h, w, true, inb, 2, 0);
//		window.shift_pixels_left();
//		window.insert_right_col(h, w, true, inb, 2, 1);
////		assertArrayEquals({0, 4, 5, 0, 7, 8, 0, 0, 0}, window.get_buf());
//
//		window.shift_pixels_left();
//		window.insert_right_col(h, w, true, inb, 2, 2);
////		assertArrayEquals({4, 5, 6, 7, 8, 9, 0, 0, 0}, window.get_buf());
//
//		window.shift_pixels_left();
//		window.insert_right_col(h, w, true, inb, 2, 3);
////		assertArrayEquals({5, 6, 0, 8, 9, 0, 0, 0, 0}, window.get_buf());
	}

//	static void test_linebuffer() {
//		LineBuffer<8, 3, int, hls::vector<int, 3 * 3>> linebuf;
//		linebuf.reset(6);
//
//		print_vector(linebuf.get_window());
//		assertArrayEquals({0, 0, 0, 0, 0, 0, 0, 0, 0}, linebuf.get_window());
//		// line 0
//		linebuf.insert_linebuf(0);
//		linebuf.insert_linebuf(0);
//		linebuf.insert_linebuf(0);
//		linebuf.insert_linebuf(0);
//		linebuf.insert_linebuf(0);
//		linebuf.insert_linebuf(0);
//		// line 1
//		linebuf.insert_linebuf(0);
//		linebuf.insert_linebuf(1);
//		linebuf.insert_linebuf(2);
//		linebuf.insert_linebuf(3);
//		linebuf.insert_linebuf(4);
//		linebuf.insert_linebuf(0);
//		// line 2
//		linebuf.slide_window(0);
//		linebuf.slide_window(5);
//		linebuf.slide_window(6);
//		assertArrayEquals({0, 0, 0, 0, 1, 2, 0, 5, 6}, linebuf.get_window());
//		linebuf.slide_window(7);
//		assertArrayEquals({0, 0, 0, 1, 2, 3, 5, 6, 7}, linebuf.get_window());
//		linebuf.slide_window(8);
//		linebuf.slide_window(0);
//		assertArrayEquals({0, 0, 0, 3, 4, 0, 7, 8, 0}, linebuf.get_window());
//		// line 3
//		linebuf.slide_window(0);
//		linebuf.slide_window(9);
//		linebuf.slide_window(10);
//		linebuf.slide_window(11);
//		linebuf.slide_window(12);
//		linebuf.slide_window(0);
//		// line 4
//		linebuf.slide_window(0);
//		linebuf.slide_window(13);
//		linebuf.slide_window(14);
//		linebuf.slide_window(15);
//		linebuf.slide_window(16);
//		linebuf.slide_window(0);
//		// line 5
//		linebuf.slide_window(0);
//		linebuf.slide_window(17);
//		linebuf.slide_window(18);
//		linebuf.slide_window(19);
//		linebuf.slide_window(20);
//		linebuf.slide_window(0);
//		// line 6
//		linebuf.slide_window(0);
//		linebuf.slide_window(0);
//		linebuf.slide_window(0);
//		assertArrayEquals({0, 13, 14, 0, 17, 18, 0, 0, 0}, linebuf.get_window());
//		linebuf.slide_window(0);
//		linebuf.slide_window(0);
//		linebuf.slide_window(0);
//		assertArrayEquals({15, 16, 0, 19, 20, 0, 0, 0, 0}, linebuf.get_window());
//	}
};
