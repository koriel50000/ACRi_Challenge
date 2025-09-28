#pragma once

#include <stdint.h>
#include "hls_stream.h"

extern "C" {
void kernel(
  hls::stream<uint64_t>& ins,
  int out[16]
);
}
