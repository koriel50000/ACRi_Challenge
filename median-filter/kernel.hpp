#pragma once

#include <cstdint>

const int WIDTH = 128;
const int HEIGHT = 32;

extern "C" {
void kernel(
  const uint8_t in[WIDTH * HEIGHT],
  uint8_t out[WIDTH * HEIGHT]
);
}
