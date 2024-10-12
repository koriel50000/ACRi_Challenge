#include <random>

#include "kernel.hpp"

int main(int argc, char** argv)
{
  // Randomize input vector
  //std::random_device seed;
  auto seed = []() { return 1; };
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(-1000.0, 1000.0);

  float table[256];
  uint8_t in[1024];

  for (int i = 0; i < 256; i++) {
    table[i] = dist(engine);
  }

  const int size = 1024;
  for (int i = 0; i < size; i++) {
    in[i] = engine() & 0xff;
  }

  float out[1024];

  kernel(table, in, size, out);

  // Check
  bool pass = true;
  for (int i = 0; i < size; i++) {
    if (out[i] != table[in[i]]) pass = false;
  }
  if (!pass) return EXIT_FAILURE;
}
