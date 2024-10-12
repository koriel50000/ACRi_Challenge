#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#define I3(i) int_t<4,4>({ (i >> 0) & 7, (i >> 3) & 7, (i >> 6) & 7, (i >> 9) & 7, })

#define C(i) int_t<4,1>({ i })
#define C4(i) int_t<4,4>({ (i >> 0) & 0xf, (i >> 4) & 0xf, (i >> 8) & 0xf, (i >> 12) & 0xf, })
#define C16(i) int_t<4,16>({ (i >> 0) & 0xf, (i >> 4) & 0xf, (i >> 8) & 0xf, (i >> 12) & 0xf, (i >> 16) & 0xf, (i >> 20) & 0xf, (i >> 24) & 0xf, (i >> 28) & 0xf, (i >> 32) & 0xf, (i >> 36) & 0xf, (i >> 40) & 0xf, (i >> 44) & 0xf, (i >> 48) & 0xf, (i >> 52) & 0xf, (i >> 56) & 0xf, (i >> 60) & 0xf, })
#define C64(i0,i1,i2,i3) int_t<4,64>({ (i0 >> 0) & 0xf, (i0 >> 4) & 0xf, (i0 >> 8) & 0xf, (i0 >> 12) & 0xf, (i0 >> 16) & 0xf, (i0 >> 20) & 0xf, (i0 >> 24) & 0xf, (i0 >> 28) & 0xf, (i0 >> 32) & 0xf, (i0 >> 36) & 0xf, (i0 >> 40) & 0xf, (i0 >> 44) & 0xf, (i0 >> 48) & 0xf, (i0 >> 52) & 0xf, (i0 >> 56) & 0xf, (i0 >> 60) & 0xf, (i1 >> 0) & 0xf, (i1 >> 4) & 0xf, (i1 >> 8) & 0xf, (i1 >> 12) & 0xf, (i1 >> 16) & 0xf, (i1 >> 20) & 0xf, (i1 >> 24) & 0xf, (i1 >> 28) & 0xf, (i1 >> 32) & 0xf, (i1 >> 36) & 0xf, (i1 >> 40) & 0xf, (i1 >> 44) & 0xf, (i1 >> 48) & 0xf, (i1 >> 52) & 0xf, (i1 >> 56) & 0xf, (i1 >> 60) & 0xf, (i2 >> 0) & 0xf, (i2 >> 4) & 0xf, (i2 >> 8) & 0xf, (i2 >> 12) & 0xf, (i2 >> 16) & 0xf, (i2 >> 20) & 0xf, (i2 >> 24) & 0xf, (i2 >> 28) & 0xf, (i2 >> 32) & 0xf, (i2 >> 36) & 0xf, (i2 >> 40) & 0xf, (i2 >> 44) & 0xf, (i2 >> 48) & 0xf, (i2 >> 52) & 0xf, (i2 >> 56) & 0xf, (i2 >> 60) & 0xf, (i3 >> 0) & 0xf, (i3 >> 4) & 0xf, (i3 >> 8) & 0xf, (i3 >> 12) & 0xf, (i3 >> 16) & 0xf, (i3 >> 20) & 0xf, (i3 >> 24) & 0xf, (i3 >> 28) & 0xf, (i3 >> 32) & 0xf, (i3 >> 36) & 0xf, (i3 >> 40) & 0xf, (i3 >> 44) & 0xf, (i3 >> 48) & 0xf, (i3 >> 52) & 0xf, (i3 >> 56) & 0xf, (i3 >> 60) & 0xf, })

template <int W, int N>
using int_t = hls::vector<ap_uint<W>, N>;
template <typename T>
using fifo = hls::stream<T>;

template <typename T, int N>
using win_t = hls::vector<T, N>;

using int4_t = ap_uint<4>;

extern "C" {
void kernel(
  int in[640 * 640],
  int out_obj_8[6400 * 1],
  int out_cls_8[6400 * 1],
  int out_bbox_8[6400 * 4],
  int out_kps_8[6400 * 10],
  int out_obj_16[1600 * 1],
  int out_cls_16[1600 * 1],
  int out_bbox_16[1600 * 4],
  int out_kps_16[1600 * 10],
  int out_obj_32[400 * 1],
  int out_cls_32[400 * 1],
  int out_bbox_32[400 * 4],
  int out_kps_32[400 * 10]
);
}
