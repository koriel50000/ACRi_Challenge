#include <algorithm>
#include <random>

#include "kernel.hpp"

int main(int argc, char** argv)
{
  // Randomize input vector
  std::default_random_engine engine(1);
  std::uniform_real_distribution<> dist(-1000, 1000);

  float in[SIZE];
  for (int i = 0; i < SIZE; i++) {
    in[i] = dist(engine);
  }

  float out_hls[SIZE];

  kernel(in, out_hls);

  std::sort(in, in + SIZE, std::greater<float>());

  // Check
  bool pass = true;
  for (int i = 0; i < SIZE; i++) {
    if (out_hls[i] != in[i]) pass = false;
  }
  if (!pass) return EXIT_FAILURE;
}
