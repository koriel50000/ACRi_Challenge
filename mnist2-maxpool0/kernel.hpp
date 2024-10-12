#pragma once

extern "C" {
void kernel(
  int in[24 * 24 * 16],
  int out[12 * 12 * 16]
);
}
