#pragma once

const int SIZE = 8;

extern "C" {
void kernel(const int in[1024], int out[1024]);
}
