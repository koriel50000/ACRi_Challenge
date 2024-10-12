#pragma once

extern "C" {
void kernel(
  int in[28 * 28 * 1],
  int out[1]
);
}
