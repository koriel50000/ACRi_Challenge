#pragma once
#include "hls_stream.h"

extern "C" {
void kernel(
  hls::stream<float>& stream_data,
  hls::stream<bool>& stream_end,
  float* output
);
}
