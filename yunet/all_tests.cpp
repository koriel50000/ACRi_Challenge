#include <iostream>
#include "xunit.hpp"
#include "bitarith_test.hpp"
#include "window_buffer_test.hpp"
#include "predictions_test.hpp"

int main_bak(int argc, char** argv)
{
	std::vector<TestEntry> tests = {
		TEST(BitArithTest::test_mul),
		TEST(BitArithTest::test_muladd),
		TEST(BitArithTest::test_batch_norm_relu),
		TEST(BitArithTest::test_batch_norm),
		TEST(WindowBufferTest::test_window),
//		TEST(WindowBufferTest::test_linebuffer),
		TEST(PredictionsTest::test_is_iou_suppressed),
		TEST(PredictionsTest::test_push_raw_pred),
		TEST(PredictionsTest::test_generate_detections),
		TEST(PredictionsTest::test_get_bboxes),
	};

	TestRunner runner;
	runner.runAllTests(tests);

	return 0;
}
