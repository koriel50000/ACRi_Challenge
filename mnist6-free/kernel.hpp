#pragma once
#include <stdint.h>
#include "hls_stream.h"

extern "C" {
void kernel_inner(
  hls::stream<uint64_t>& ins,
  int out[1]
);
}
