#pragma once
#include "hls_stream.h"

extern "C" {
void kernel(
  hls::stream<long>& ins,
  int out[1]
);
}
