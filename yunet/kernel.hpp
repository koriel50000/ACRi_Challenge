#pragma once

extern "C" {
void kernel(
  int in[160 * 160],
  int out[16]
);
}
