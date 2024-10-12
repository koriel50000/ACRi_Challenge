#pragma once

const int SIZE = 16384;

extern "C" {
void kernel(
  const float in[SIZE],
  float out[5]
);
}
