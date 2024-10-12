#pragma once

#include <cstdint>

extern "C" {
void kernel(
  int in[28 * 28 * 1],
  int weight[16 * 5 * 5 * 1],
  int threshold[3],
  int out[24 * 24 * 16]
);
}
