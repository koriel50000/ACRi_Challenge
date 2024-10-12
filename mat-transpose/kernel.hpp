#pragma once

const int SIZE = 256;

extern "C" {
void kernel(const float in[SIZE * SIZE], float out[SIZE * SIZE]);
}
