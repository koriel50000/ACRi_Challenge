#pragma once

#include <cstdint>

const int SIZE = 1024;

extern "C" {
void kernel(
  uint8_t data[SIZE],
  uint64_t code[256],
  uint64_t code_size[256],
  uint8_t out[SIZE*8]
);
}
