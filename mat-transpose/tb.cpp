#include <vector>
#include <random>

#include "kernel.hpp"

int main(int argc, char** argv)
{
  // Randomize input vector
  auto seed = []() { return 1; };
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(-1000.0, 1000.0);

  std::vector<float> in(SIZE * SIZE), out(SIZE * SIZE);

  for (int i = 0; i < SIZE * SIZE; i++) {
    in[i] = dist(engine);
  }

  kernel(in.data(), out.data());

  // Check
  bool pass = true;
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      if (in[i * SIZE + j] != out[j * SIZE + i]) pass = false;
    }
  }
  if (!pass) return EXIT_FAILURE;
}
