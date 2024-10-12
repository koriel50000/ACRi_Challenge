#pragma once

const int TEXT_SIZE = 16384;
const int PATTERN_SIZE = 32;
const int MAX_MATCH = 16;

extern "C" {
void kernel(
  const char in_text[TEXT_SIZE],
  const char in_pat[PATTERN_SIZE],
  int out_match_pos[MAX_MATCH],
  int* out_match_num
);
}
