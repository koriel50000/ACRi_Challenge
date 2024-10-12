#pragma once

extern "C" {
void kernel(
  int in[12 * 12 * 16],
  int weight[16 * 5 * 5 * 16],
  int threshold[3],
  int out[4 * 4 * 16]
);
}
