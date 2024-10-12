#pragma once

// https://forums.xilinx.com/t5/High-Level-Synthesis-HLS/Vivado-2015-3-HLS-Bug-gmp-h/m-p/1034913/highlight/true#M18404
#include <gmp.h>
#define __gmp_const const

#include "ap_int.h"

extern "C" {
void kernel(
  const ap_uint<128> plain,
  const ap_uint<128> e,
  const ap_uint<256> n,
  ap_uint<256>* encrypted
);
}
