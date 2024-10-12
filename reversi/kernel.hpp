#pragma once

#include <at_int.h>

const int ROWS = 8;
const int COLUMNS = 8;
const int CHANNELS = 2;

extern "C" {
void kernel(
  const bool reset,
  const ap_int<6> current_move,
  float state_buffer[ROWS * COLUMNS * CHANNELS]
);
}
