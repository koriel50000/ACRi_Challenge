#pragma once
#include <cstdint>
#include "hls_vector.h"

typedef hls::vector<uint32_t, 16> block_t;
typedef hls::vector<uint32_t, 8> hash_t;

extern "C" {
void kernel(
  const block_t input[1024],
  const int size,
  hash_t* output
);
}