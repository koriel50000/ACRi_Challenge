#include <iostream>
#include "xunit.hpp"
#include "bitarith_test.hpp"
#include "window_buffer_test.hpp"

int main(int argc, char** argv)
{
	std::vector<TestEntry> tests = {
		TEST(BitArithTest::test_mul),
		TEST(BitArithTest::test_muladd),
		TEST(BitArithTest::test_batch_norm_relu),
		TEST(BitArithTest::test_batch_norm),
		TEST(WindowBufferTest::test_window),
//		TEST(WindowBufferTest::test_linebuffer),
	};

	TestRunner runner;
	runner.runAllTests(tests);

	return 1;
}
