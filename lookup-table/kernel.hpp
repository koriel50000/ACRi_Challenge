#pragma once

#include <cstdint>

extern "C" {
void kernel(
  const float table[256],
  const uint8_t in[1024],
  const int size,
  float out[1024]
);
}
