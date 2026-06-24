#include <iostream>
#include "predictions.hpp"
#include "layers.hpp"
#include "xunit.hpp"
#include "output.hpp"

using namespace Assertions;

class PredictionsTest
{
private:
	static void write_testdata_stream(fifo<data_t>& ins) {
		// YuNet_Head cls ConvDPUnit
		// YuNet_Head stride8
		for (int i = 0; i < 20 * 20; i++) {
			ins.write(cls_scores_stride8[i]);
		}
		// YuNet_Head stride16
		for (int i = 0; i < 10 * 10; i++) {
			ins.write(cls_scores_stride16[i]);
		}
		// YuNet_Head stride32
		for (int i = 0; i < 5 * 5; i++) {
			ins.write(cls_scores_stride32[i]);
		}

		// YuNet_Head bbox ConvDPUnit
		// YuNet_Head stride8
		for (int i = 0; i < 20 * 20; i++) {
			ins.write(bbox_preds_stride8[i]);
		}
		// YuNet_Head stride16
		for (int i = 0; i < 10 * 10; i++) {
			ins.write(bbox_preds_stride16[i]);
		}
		// YuNet_Head stride32
		for (int i = 0; i < 5 * 5; i++) {
			ins.write(bbox_preds_stride32[i]);
		}

		// YuNet_Head obj ConvDPUnit
		// YuNet_Head stride8
		for (int i = 0; i < 20 * 20; i++) {
			ins.write(objectnesses_stride8[i]);
		}
		// YuNet_Head stride16
		for (int i = 0; i < 10 * 10; i++) {
			ins.write(objectnesses_stride16[i]);
		}
		// YuNet_Head stride32
		for (int i = 0; i < 5 * 5; i++) {
			ins.write(objectnesses_stride32[i]);
		}

		// YuNet_Head kps ConvDPUnit
		// YuNet_Head stride8
		for (int i = 0; i < 20 * 20; i++) {
			ins.write(kps_preds_stride8[i]);
		}
		// YuNet_Head stride16
		for (int i = 0; i < 10 * 10; i++) {
			ins.write(kps_preds_stride16[i]);
		}
		// YuNet_Head stride32
		for (int i = 0; i < 5 * 5; i++) {
			ins.write(kps_preds_stride32[i]);
		}		
	}

	static void print_block(const char* title, uint8_t preds[525][16], int feat_begin, int feat_end) {
		printf("%s\n", title);

		// stride 8 (20x20)
		printf("8:\n");
		for (int f = feat_begin; f <= feat_end; f++) {
			for (int i = 0; i < 400; i++) {
				if (i % 20 == 0) printf("   [");
				printf("%2d, ", preds[i][f]);
				if (i % 20 == 19) printf("]\n");
			}
			printf("-\n\n");
		}

		// stride 16 (10x10)
		printf("16:\n");
		for (int f = feat_begin; f <= feat_end; f++) {
			for (int i = 400; i < 500; i++) {
				int idx = i - 400;
				if (idx % 10 == 0) printf("   [");
				printf("%2d, ", preds[i][f]);
				if (idx % 10 == 9) printf("]\n");
			}
			printf("-\n\n");
		}

		// stride 32 (5x5)
		printf("32:\n");
		for (int f = feat_begin; f <= feat_end; f++) {
			for (int i = 500; i < 525; i++) {
				int idx = i - 500;
				if (idx % 5 == 0) printf("   [");
				printf("%2d, ", preds[i][f]);
				if (idx % 5 == 4) printf("]\n");
			}
			printf("-\n\n");
		}
	}
public:
	static void test_is_iou_suppressed()
	{
		Predictions<20, 20, 10, data_t> preds;

		Detection box1 = { 10, 10, 20, 20, 0 };
		Detection box2 = { 11, 11, 21, 21, 0 };
		assertTrue(preds.is_iou_suppressed(box1, box2));

		Detection box3 = { 10, 10, 20, 20, 0 };
		Detection box4 = { 7, 7, 17, 17, 0 };
		assertFalse(preds.is_iou_suppressed(box3, box4));

		Detection box5 = { 10, 10, 30, 30, 0 };
		Detection box6 = { 15, 15, 25, 25, 0 };
		assertTrue(preds.is_iou_suppressed(box1, box2));

		Detection box7 = { 10, 10, 20, 20, 0 };
		Detection box8 = { 30, 30, 40, 40, 0 };
		assertFalse(preds.is_iou_suppressed(box7, box8));
	}

	static void test_push_raw_pred() {
		fifo<data_t> ins;
		write_testdata_stream(ins);

		Predictions<20, 20, 10, data_t> preds;

		// YuNet_Head cls ConvDPUnit
		// YuNet_Head stride8
		preds.push_raw_pred(20, 20, 1, ins);
		assertEquals(400, preds.pred_ptr);
		assertEquals(1, preds.feat_index);
		assertEquals(0, preds.preds[0][0]);
		assertEquals(9, preds.preds[2][0]);
		// // YuNet_Head stride16
		preds.push_raw_pred(10, 10, 1, ins);
		assertEquals(500, preds.pred_ptr);
		assertEquals(2, preds.feat_index);
		assertEquals(9, preds.preds[400][0]);
		assertEquals(10, preds.preds[402][0]);
		// // YuNet_Head stride32
		preds.push_raw_pred(5, 5, 1, ins);
		assertEquals(0, preds.pred_ptr);
		assertEquals(3, preds.feat_index);
		assertEquals(10, preds.preds[500][0]);
		assertEquals(12, preds.preds[501][0]);

		// // YuNet_Head bbox ConvDPUnit
		// // YuNet_Head stride8
		preds.push_raw_pred(20, 20, 4, ins);
		assertEquals(400, preds.pred_ptr);
		assertEquals(4, preds.feat_index);
		assertEquals(9, preds.preds[0][1]);
		assertEquals(0, preds.preds[0][3]);
		assertEquals(2, preds.preds[2][2]);
		assertEquals(1, preds.preds[2][4]);
		// // YuNet_Head stride16
		preds.push_raw_pred(10, 10, 4, ins);
		assertEquals(500, preds.pred_ptr);
		assertEquals(5, preds.feat_index);
		assertEquals(12, preds.preds[400][1]);
		assertEquals(12, preds.preds[400][3]);
		assertEquals(10, preds.preds[401][2]);
		assertEquals(11, preds.preds[401][4]);
		// // YuNet_Head stride32
		preds.push_raw_pred(5, 5, 4, ins);
		assertEquals(0, preds.pred_ptr);
		assertEquals(6, preds.feat_index);
		assertEquals(10, preds.preds[500][1]);
		assertEquals(9, preds.preds[500][3]);
		assertEquals(1, preds.preds[501][2]);
		assertEquals(0, preds.preds[501][4]);

		// YuNet_Head obj ConvDPUnit
		// YuNet_Head stride8
		preds.push_raw_pred(20, 20, 1, ins);
		assertEquals(400, preds.pred_ptr);
		assertEquals(7, preds.feat_index);
		assertEquals(9, preds.preds[0][5]);
		assertEquals(0, preds.preds[2][5]);
		// YuNet_Head stride16
		preds.push_raw_pred(10, 10, 1, ins);
		assertEquals(500, preds.pred_ptr);
		assertEquals(8, preds.feat_index);
		assertEquals(9, preds.preds[400][5]);
		assertEquals(10, preds.preds[401][5]);
		// YuNet_Head stride32
		preds.push_raw_pred(5, 5, 1, ins);
		assertEquals(0, preds.pred_ptr);
		assertEquals(9, preds.feat_index);
		assertEquals(11, preds.preds[500][5]);
		assertEquals(9, preds.preds[507][5]);

		// YuNet_Head kps ConvDPUnit
		// YuNet_Head stride8
		preds.push_raw_pred(20, 20, 10, ins);
		assertEquals(400, preds.pred_ptr);
		assertEquals(10, preds.feat_index);
		assertEquals(9, preds.preds[0][6]);
		assertEquals(9, preds.preds[0][7]);
		assertEquals(0, preds.preds[1][14]);
		assertEquals(5, preds.preds[1][15]);
		// YuNet_Head stride16
		preds.push_raw_pred(10, 10, 10, ins);
		assertEquals(500, preds.pred_ptr);
		assertEquals(11, preds.feat_index);
		assertEquals(9, preds.preds[400][6]);
		assertEquals(0, preds.preds[400][7]);
		assertEquals(10, preds.preds[401][12]);
		assertEquals(2, preds.preds[401][13]);
		// YuNet_Head stride32
		preds.push_raw_pred(5, 5, 10, ins);
		assertEquals(0, preds.pred_ptr);
		assertEquals(12, preds.feat_index);
		assertEquals(0, preds.preds[500][12]);
		assertEquals(1, preds.preds[500][13]);
		assertEquals(9, preds.preds[501][14]);
		assertEquals(2, preds.preds[501][15]);

		// print_block("cls",  preds.preds, 0, 0);
		// print_block("bbox", preds.preds, 1, 4);
		// print_block("obj",  preds.preds, 5, 5);
		// print_block("kps",  preds.preds, 6, 15);
	}

	static void test_generate_detections() {
		fifo<data_t> ins;
		write_testdata_stream(ins);

		Predictions<20, 20, 10, data_t> preds;

		// YuNet_Head cls ConvDPUnit
		preds.push_raw_pred(20, 20, 1, ins);
		preds.push_raw_pred(10, 10, 1, ins);
		preds.push_raw_pred(5, 5, 1, ins);
		// // YuNet_Head bbox ConvDPUnit
		preds.push_raw_pred(20, 20, 4, ins);
		preds.push_raw_pred(10, 10, 4, ins);
		preds.push_raw_pred(5, 5, 4, ins);
		// YuNet_Head obj ConvDPUnit
		preds.push_raw_pred(20, 20, 1, ins);
		preds.push_raw_pred(10, 10, 1, ins);
		preds.push_raw_pred(5, 5, 1, ins);
		// YuNet_Head kps ConvDPUnit
		preds.push_raw_pred(20, 20, 10, ins);
		preds.push_raw_pred(10, 10, 10, ins);
		preds.push_raw_pred(5, 5, 10, ins);

		uint8_t size = preds.generate_detections();
		assertEquals(20, size);
		Detection& detect1 = preds.detects[0];
		assertEquals(13, detect1.x1);
		assertEquals(33, detect1.y1);
		assertEquals(35, detect1.x2);
		assertEquals(63, detect1.y2);
		assertEquals(35739, detect1.score);
		assertEquals(19, detect1.kps[0]);
		assertEquals(56, detect1.kps[9]);
		Detection& detect2 = preds.detects[19];
		assertEquals(11, detect2.x1);
		assertEquals(121, detect2.y1);
		assertEquals(47, detect2.x2);
		assertEquals(157, detect2.y2);
		assertEquals(45942, detect2.score);
		assertEquals(24, detect2.kps[0]);
		assertEquals(145, detect2.kps[9]);

		// for (int i = 0; i < 20; i++) {
		// 	Detection& detect = preds.get_detection(i);
		// 	printf("%d, %d, %d, %d, ", detect.x1, detect.y1, detect.x2, detect.y2);			
		// 	// for (int j = 0; j < 10; j++) {
		// 	// 	printf("%d, ", detect.kps[j]);
		// 	// }
		// 	printf("\n");
		// }
		// printf("\n");
	}

	static void test_get_bboxes() {
		fifo<data_t> ins;
		write_testdata_stream(ins);

		Predictions<20, 20, 10, data_t> preds;

		// YuNet_Head cls ConvDPUnit
		preds.push_raw_pred(20, 20, 1, ins);
		preds.push_raw_pred(10, 10, 1, ins);
		preds.push_raw_pred(5, 5, 1, ins);
		// // YuNet_Head bbox ConvDPUnit
		preds.push_raw_pred(20, 20, 4, ins);
		preds.push_raw_pred(10, 10, 4, ins);
		preds.push_raw_pred(5, 5, 4, ins);
		// YuNet_Head obj ConvDPUnit
		preds.push_raw_pred(20, 20, 1, ins);
		preds.push_raw_pred(10, 10, 1, ins);
		preds.push_raw_pred(5, 5, 1, ins);
		// YuNet_Head kps ConvDPUnit
		preds.push_raw_pred(20, 20, 10, ins);
		preds.push_raw_pred(10, 10, 10, ins);
		preds.push_raw_pred(5, 5, 10, ins);

		uint8_t size = preds.get_bboxes();
		assertEquals(6, size);
		Detection& detect0 = preds.get_detection(0);
		Detection& detect1 = preds.get_detection(1);
		Detection& detect2 = preds.get_detection(2);
		Detection& detect3 = preds.get_detection(3);
		Detection& detect4 = preds.get_detection(4);
		Detection& detect5 = preds.get_detection(5);

		assertEquals(48, detect0.x1);
		assertEquals(36, detect0.y1);
		assertEquals(84, detect0.x2);
		assertEquals(84, detect0.y2);
		assertEquals(49153, detect0.score);
		assertArrayEquals<uint8_t>({56, 53, 67, 51, 61, 59, 57, 66, 70, 66}, detect0.kps);
		assertEquals(110, detect1.x1);
		assertEquals(65, detect1.y1);
		assertEquals(146, detect1.x2);
		assertEquals(113, detect1.y2);
		assertEquals(45942, detect1.score);
		assertArrayEquals<uint8_t>({126, 83, 138, 83, 134, 89, 129, 98, 138, 98}, detect1.kps);
		assertEquals(11, detect2.x1);
		assertEquals(121, detect2.y1);
		assertEquals(47, detect2.x2);
		assertEquals(157, detect2.y2);
		assertEquals(45942, detect2.score);
		assertArrayEquals<uint8_t>({24, 136, 35, 136, 30, 139, 24, 147, 35, 145}, detect2.kps);
		assertEquals(13, detect3.x1);
		assertEquals(33, detect3.y1);
		assertEquals(35, detect3.x2);
		assertEquals(63, detect3.y2);
		assertEquals(42237, detect3.score);
		assertArrayEquals<uint8_t>({19, 43, 25, 43, 22, 49, 19, 54, 27, 54}, detect3.kps);
		assertEquals(131, detect4.x1);
		assertEquals(38, detect4.y1);
		assertEquals(157, detect4.x2);
		assertEquals(68, detect4.y2);
		assertEquals(38874, detect4.score);
		assertArrayEquals<uint8_t>({144, 49, 150, 49, 150, 53, 145, 59, 154, 59}, detect4.kps);
		assertEquals(70, detect5.x1);
		assertEquals(97, detect5.y1);
		assertEquals(106, detect5.x2);
		assertEquals(145, detect5.y2);
		assertEquals(35739, detect5.score);
		assertArrayEquals<uint8_t>({88, 115, 96, 115, 93, 121, 88, 130, 96, 130}, detect5.kps);
	}
};
