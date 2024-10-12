#include <cassert>
#include <vector>
#include <random>

#include "kernel.hpp"

int main(int argc, char** argv)
{
  // Randomize input vector
  //std::random_device seed;
  auto seed = []() { return 1; };
  std::default_random_engine engine(seed());
  std::uniform_int_distribution<int> dist(-1000, 1000);

  int in[SIZE], out[SIZE];

  for (int i=0; i<SIZE; i++) {
      in[i] = dist(engine);
  }

  kernel(in, out);

  // Check (not check the actual value in this public-version)
  for (int i=1; i<SIZE; i++) {
    assert(out[i-1] <= out[i]);
  }
}
