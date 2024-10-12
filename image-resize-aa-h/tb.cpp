#include <fstream>
#include <random>

#include "kernel.hpp"

void golden(
  const uint8_t in[1024],
  const uint32_t out_size,
  uint8_t out[1024]
) {
  uint32_t in_sum = 0;
  uint32_t in_num = 0;
  uint32_t out_idx = 0;
  for (int i = 0; i < 1024; i++) {
    uint8_t in_pix = in[i];
    uint32_t rest = 1024 - in_num;
    if (rest <= out_size) {
      out[out_idx++] = (in_sum + in_pix * rest + 512) / 1024;
      in_sum = in_pix * (out_size - rest);
      in_num = (out_size - rest);
    } else {
      in_sum += in_pix * out_size;
      in_num += out_size;
    }
  }
}

int main(int argc, char** argv)
{
  // Randomize input vector
  auto seed = []() { return 1; };
  std::default_random_engine engine(seed());
  std::uniform_int_distribution<> dist(0, 255);

  uint8_t in[1024], out_golden[1024], out_hls[1024];

  const int size = 1024;
  for (int i = 0; i < size; i++) {
    in[i] = dist(engine);
  }

  // Check
  bool pass = true;

  for (auto out_size : { 512, 300, 800, 64, 1023 }) {
    // Golden
    golden(in, out_size, out_golden);

    // HLS
    kernel(in, out_size, out_hls);

    for (int i = 0; i < out_size; i++) {
      if (out_hls[i] != out_golden[i]) pass = false;
    }
  }

  if (!pass) return EXIT_FAILURE;
}
