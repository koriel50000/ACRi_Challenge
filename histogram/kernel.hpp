#pragma once

#include <cstdint>

extern "C" {
void kernel(
  const uint8_t in[8192],
  const int size,
  uint16_t hist[256]
);
}
