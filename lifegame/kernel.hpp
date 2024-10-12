#pragma once

#include <ap_int.h>

extern "C" {
void kernel(const int generation, const ap_uint<1> in[64], ap_uint<1> out[64]);
}
