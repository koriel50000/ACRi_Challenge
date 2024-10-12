#pragma once

extern "C" {
void kernel(
  int in[256],
  int weight[10 * 256],
  int out[10]
);
}
