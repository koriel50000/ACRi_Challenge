#include <iostream>
#include "bitarith.hpp"
#include "xunit.hpp"

using namespace Assertions;

class BitArithTest
{
public:
	static void test_mul()
	{
		assertEquals(0, mul(0b0000, 0b0000));
		assertEquals(0, mul(0b0111, 0b0000));
		assertEquals(0, mul(0b0000, 0b1111));
		assertEquals(1 * 1, mul(0b0001, 0b0001));
		assertEquals(3 * 4, mul(0b0011, 0b0011));
		assertEquals(12 * 64, mul(0b0111, 0b0111));
		assertEquals((-1) * (-1), mul(0b1001, 0b1001));
		assertEquals((-3) * (-4), mul(0b1011, 0b1011));
		assertEquals((-12) * (-64), mul(0b1111, 0b1111));
		assertEquals((-1) * 1, mul(0b1001, 0b0001));
		assertEquals((-3) * 4, mul(0b1011, 0b0011));
		assertEquals((-12) * 64, mul(0b1111, 0b0111));
		assertEquals(1 * (-1), mul(0b0001, 0b1001));
		assertEquals(3 * (-4), mul(0b0011, 0b1011));
		assertEquals(12 * (-64), mul(0b0111, 0b1111));
	}

	static void test_muladd()
	{
		int_t<4> v = int_t<4>(0x1234);
		int_t<4> w1 = int_t<4>(0x1234);
		assertEquals(49, muladd<4>(v, w1));
		int_t<4> w2 = int_t<4>(0x9abc);
		assertEquals(-49, muladd<4>(v, w2));
	}

	static void test_batch_norm_relu()
	{
		int16_t thr[7] = {10, 20, 30, 40, 50, 60, 70};

		assertEquals(0b0000, batch_norm_relu(5, thr));
		assertEquals(0b0001, batch_norm_relu(10, thr));
		assertEquals(0b0001, batch_norm_relu(15, thr));
		assertEquals(0b0010, batch_norm_relu(25, thr));
		assertEquals(0b0011, batch_norm_relu(35, thr));
		assertEquals(0b0100, batch_norm_relu(45, thr));
		assertEquals(0b0101, batch_norm_relu(55, thr));
		assertEquals(0b0110, batch_norm_relu(65, thr));
		assertEquals(0b0111, batch_norm_relu(100, thr));
	}


	static void test_batch_norm()
	{
		int16_t thr[14] = {-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70};

		assertEquals(0b1111, batch_norm(-65, thr));
		assertEquals(0b1110, batch_norm(-60, thr));
		assertEquals(0b1110, batch_norm(-55, thr));
		assertEquals(0b1101, batch_norm(-45, thr));
		assertEquals(0b1100, batch_norm(-35, thr));
		assertEquals(0b1011, batch_norm(-25, thr));
		assertEquals(0b1010, batch_norm(-15, thr));
		assertEquals(0b1001, batch_norm(-5, thr));
		assertEquals(0b0000, batch_norm(5, thr));
		assertEquals(0b0001, batch_norm(15, thr));
		assertEquals(0b0010, batch_norm(25, thr));
		assertEquals(0b0011, batch_norm(35, thr));
		assertEquals(0b0100, batch_norm(45, thr));
		assertEquals(0b0101, batch_norm(55, thr));
		assertEquals(0b0110, batch_norm(65, thr));
		assertEquals(0b0111, batch_norm(100, thr));
	}
};
