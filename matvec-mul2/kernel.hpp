#pragma once

const int SIZE = 80;

extern "C" {
void kernel(
  const float in_mat[SIZE * SIZE],
  const float in_vec[SIZE],
  float out[SIZE]
);
}
