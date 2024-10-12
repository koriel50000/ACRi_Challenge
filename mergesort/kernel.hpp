#pragma once

const int SIZE = 32;

extern "C" {
void kernel(
  const float in[SIZE],
  float out[SIZE]
);
}
