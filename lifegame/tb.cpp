#include <cassert>
#include <vector>
#include <random>

#include "kernel.hpp"

int main(int argc, char** argv)
{
  ap_uint<1> in[64] = { 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 0, 0, 0, 0,
                        0, 1, 1, 1, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0 };
  ap_uint<1> out[64];

  const int generation = 32 * 10000;

  kernel(generation, in, out);

  // Check
  bool pass = true;
  for (int i = 0; i < 64; i++) {
    if (out[i] != in[i]) pass = false;
  }
  if (!pass) return EXIT_FAILURE;
}
