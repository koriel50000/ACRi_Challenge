#pragma once

const int SIZE = 1024;

extern "C" {
void kernel(const int in[SIZE], int out[SIZE]);
}
