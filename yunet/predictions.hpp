#pragma once
#include "types.hpp"

static const int NUM_PRED_TYPES = 4; // cls, bbox, obj, kps
static const int NUM_STRIDES = 3;
static const int STRIDES[] = { 8, 16, 32 };
static const int NUM_QLEVELS_4BIT = 16;

static const int NUM_PRED_CELLS = 20*20 + 10*10 + 5*5; // [(20, 20), (10, 10), (5, 5)]
static const int NUM_PRED_FEATURES = 16; // cls, bbox[4], obj, kps[10]
static const int IMAGE_SIZE = 160;
static const int MAX_DETECTIONS = 64;

static const uint8_t FEATURE_DECODE_TABLE[(NUM_PRED_TYPES + 1) * NUM_STRIDES][NUM_QLEVELS_4BIT] = {
    // cls
    { 136, 145, 154, 162, 171, 186, 199, 220, 0, 127, 118, 109, 100, 83, 68, 43 },
    { 145, 157, 168, 178, 188, 205, 218, 236, 0, 134, 122, 110, 98, 77, 58, 31 },
    { 136, 154, 171, 187, 200, 221, 235, 249, 0, 117, 99, 82, 67, 42, 25, 8 },
    // bbox
    { 7, 8, 9, 10, 12, 14, 16, 21, 0, 6, 5, 4, 3, 0, 0, 0 },
    { 19, 22, 24, 26, 28, 33, 37, 46, 0, 17, 15, 13, 10, 6, 2, 0 },
    { 54, 59, 64, 69, 74, 85, 95, 116, 0, 48, 43, 38, 33, 22, 12, 0 },
    // bbox(exp)
    { 20, 23, 27, 31, 36, 48, 64, 113, 0, 17, 15, 13, 11, 8, 6, 3 },
    { 55, 63, 72, 83, 96, 127, 168, 255, 0, 48, 41, 36, 31, 23, 18, 10 },
    { 173, 204, 239, 255, 255, 255, 255, 255, 0, 147, 125, 106, 90, 65, 47, 24 },
    // obj
    { 4, 1, 0, 0, 0, 0, 0, 0, 0, 10, 25, 55, 105, 209, 247, 255 },
    { 6, 17, 39, 82, 140, 227, 251, 255, 0, 2, 1, 0, 0, 0, 0, 0 },
    { 5, 13, 36, 82, 148, 235, 253, 255, 0, 1, 0, 0, 0, 0, 0, 0 },
    // kps
    { 5, 6, 8, 9, 11, 14, 18, 24, 0, 3, 1, 0, 0, 0, 0, 0 },
    { 14, 17, 21, 25, 28, 36, 43, 57, 0, 10, 6, 3, 0, 0, 0, 0 },
    { 29, 40, 52, 63, 74, 97, 120, 166, 0, 17, 6, 0, 0, 0, 0, 0 },
};

static const uint8_t FEATURE_OFFSET[NUM_PRED_TYPES * NUM_STRIDES] = {
    0, 0, 0,  // cls
    1, 1, 1,  // bbox
    5, 5, 5,  // obj
    6, 6, 6,  // kps
};

struct Detection {
    uint8_t  x1;
    uint8_t  y1;
    uint8_t  x2;
    uint8_t  y2;
    uint16_t score;
    uint8_t  kps[10]; // (x, y) * 5
};

template<int H, int W, int F, typename T>
class Predictions {
    friend class PredictionsTest;     

private:
    uint4_t preds[NUM_PRED_CELLS][NUM_PRED_FEATURES];
    int pred_ptr = 0;
    int feat_index = 0;
    Detection detects[MAX_DETECTIONS];

    bool is_iou_suppressed(const Detection& box1, const Detection& box2) {
        uint8_t x1 = (box1.x1 > box2.x1) ? box1.x1 : box2.x1;
        uint8_t y1 = (box1.y1 > box2.y1) ? box1.y1 : box2.y1;
        uint8_t x2 = (box1.x2 < box2.x2) ? box1.x2 : box2.x2;
        uint8_t y2 = (box1.y2 < box2.y2) ? box1.y2 : box2.y2;

        uint8_t w = (x2 > x1) ? (x2 - x1) : 0;
        uint8_t h = (y2 > y1) ? (y2 - y1) : 0;

        int inter_area = w * h;
        uint16_t area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
        uint16_t area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

        int union_area = area1 + area2 - inter_area;

        return inter_area * 569 >= (union_area << 8); // 0.45 ≒ 256 / 569 = 0.44991..
    }

    uint8_t generate_detections() {
        uint8_t size = 0;
        int ptr = 0;
        for (int i = 0; i < NUM_STRIDES; i++) {
            int stride = STRIDES[i];
            const uint8_t *cls = FEATURE_DECODE_TABLE[0 + i];
            const uint8_t *bbox = FEATURE_DECODE_TABLE[3 + i];
            const uint8_t *bboxexp = FEATURE_DECODE_TABLE[6 + i];
            const uint8_t *obj = FEATURE_DECODE_TABLE[9 + i];
            const uint8_t *kps = FEATURE_DECODE_TABLE[12 + i];
            for (int y = 0; y < IMAGE_SIZE; y += stride) {
                for (int x = 0; x < IMAGE_SIZE; x += stride) {
                    uint4_t *pred = preds[ptr++];
                    uint16_t score = cls[pred[0]] * obj[pred[5]]; // cls * obj
                    if (score & 0x8000) { // score >= 32768(=0.5)
                        Detection *detect = &detects[size++]; // 暗黙で64を超えないと想定
                        // decode bbox
                        uint8_t cx = x + bbox[pred[1]];
                        uint8_t cy = y + bbox[pred[2]];
                        uint8_t half_w = bboxexp[pred[3]] >> 1;
                        uint8_t half_h = bboxexp[pred[4]] >> 1;
                        detect->x1 = cx - half_w;
                        detect->y1 = cy - half_h;
                        detect->x2 = cx + half_w;
                        detect->y2 = cy + half_h;
                        detect->score = score;
                        // decode kps
                        detect->kps[0] = x + kps[pred[6]];
                        detect->kps[1] = y + kps[pred[7]];
                        detect->kps[2] = x + kps[pred[8]];
                        detect->kps[3] = y + kps[pred[9]];
                        detect->kps[4] = x + kps[pred[10]];
                        detect->kps[5] = y + kps[pred[11]];
                        detect->kps[6] = x + kps[pred[12]];
                        detect->kps[7] = y + kps[pred[13]];
                        detect->kps[8] = x + kps[pred[14]];
                        detect->kps[9] = y + kps[pred[15]];
                    }
                }
            }
        }
        return size;
    }
public:
    // cls8 -> cla16 -> cls32 -> bbox8 -> bbox16 -> bbox32 -> obj8 -> obj16 -> obj32 -> kps8 -> kps16 -> kps32
    void push_raw_pred(const int h, const int w, const int f, fifo<T>& ins) {
        int offset = FEATURE_OFFSET[feat_index];

        push_raw_pred_h: for (int y = 0; y < H; y++) {
            if (y < h) {
                push_raw_pred_w: for (int x = 0; x < W; x++) {
                    if (x < w) {
                        T val = ins.read();
                        push_raw_pred_f: for (int z = 0; z < F; z++) {
                            if (z < f) {
                                preds[pred_ptr][offset + z] = val[z];
                            }
                        }
                        pred_ptr++;
                    }
                }
            }
        }
        if (pred_ptr == NUM_PRED_CELLS) {
            pred_ptr = 0;
        }
        feat_index++;
    }

    uint8_t get_bboxes() {
        uint8_t size = generate_detections();

        uint8_t detect_size = 0;        
        for (int j = 0; j < MAX_DETECTIONS; j++) {
            if (j < size) {
                // sort
                uint8_t max_index = j;
                uint16_t max_score = detects[j].score;
                for (int i = j + 1; i < MAX_DETECTIONS; i++) {
                    if (i < size) {
                        if (detects[i].score > max_score) {
                            max_index = i;
                            max_score = detects[i].score;
                        }
                    }
                }
                if (max_score == 0) break;

                Detection tmp = detects[j];
                detects[j] = detects[max_index];
                detects[max_index] = tmp;
                detect_size++;

                // nms
                for (int i = j + 1; i < MAX_DETECTIONS; i++) {
                    if (i < size) {
                        if (detects[i].score != 0) {
                            if (is_iou_suppressed(detects[j], detects[i])) {
                                detects[i].score = 0; // 除外
                            }
                        }
                    }                    
                }                
            }
        }
        return detect_size;
    }

    Detection& get_detection(int i) {
        return detects[i];
    }
};
