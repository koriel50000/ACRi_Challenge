#include <random>
#include <vector>

#include "kernel.hpp"

int main(int argc, char** argv)
{
  // Randomize input vector
  std::default_random_engine engine(1);
  std::uniform_int_distribution<> dist(0, 255);

  std::vector<uint16_t> hist_hls(256, 0), hist_golden(256, 0);

  uint8_t in[8192];
  const int size = 8192;
  for (int i = 0; i < size; i++) {
    in[i] = dist(engine);
    hist_golden[in[i]]++;
  }

  kernel(in, size, hist_hls.data());

  // Check
  bool pass = true;
  for (int i = 0; i < 256; i++) {
    if (hist_hls[i] != hist_golden[i]) pass = false;
  }
  if (!pass) return EXIT_FAILURE;
}