#pragma once

#include <cstdint>

extern "C" {
void kernel(
  const uint8_t in[1024],
  const uint32_t out_size,
  uint8_t out[1024]
);
}
