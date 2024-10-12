#pragma once

extern "C" {
void kernel(
  int in[28 * 28 * 1],
  int conv0_weight[16 * 5 * 5 * 1],
  int threshold0[3],
  int conv1_weight[16 * 5 * 5 * 16],
  int threshold1[3],
  int matmul0_weight[256 * 10],
  int out[10]
);
}
