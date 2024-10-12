#pragma once

#include <cstdint>

extern "C" {
void kernel(uint32_t in, int16_t* out);
}
