#pragma once

#define MAX_OUTPUT_SIZE (256*256)

extern "C" {
void kernel(
  const int width,
  const int height,
  const int max_iter,
  const float start_x,
  const float start_y,
  const float step_x,
  const float step_y,
  char output[MAX_OUTPUT_SIZE]
);
}
