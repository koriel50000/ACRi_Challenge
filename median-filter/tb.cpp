#include <algorithm>
#include <random>
#include <vector>

#include "kernel.hpp"

void reference(
  const std::vector<uint8_t>& in,
  std::vector<uint8_t>& out
) {
  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      // Collect neighbor pixels
      uint8_t neighbors[3 * 3];
      uint8_t neighbor_count = 0;
      for (int ny = -1; ny <= 1; ny++) {
        for (int nx = -1; nx <= 1; nx++) {
          if (0 <= x + nx && x + nx < WIDTH && 0 <= y + ny && y + ny < HEIGHT) {
            neighbors[neighbor_count++] = in.at((y + ny) * WIDTH + (x + nx));
          }
        }
      }
      // Sort
      std::sort(neighbors, neighbors + neighbor_count);
      // Output
      out.at(y * WIDTH + x) = neighbors[neighbor_count / 2];
    }
  }
}

int main(int argc, char** argv)
{
  // Randomize input vector
  std::default_random_engine engine(1);
  std::uniform_int_distribution<> dist(0, 255);

  const int SIZE = WIDTH * HEIGHT;
  std::vector<uint8_t> in(SIZE), out_ref(SIZE), out_hls(SIZE);

  for (int i = 0; i < SIZE; i++) {
    in[i] = dist(engine);
  }

  // Reference
  reference(in, out_ref);

  // HLS
  kernel(in.data(), out_hls.data());

  // Check
  bool pass = true;
  for (int i = 0; i < SIZE; i++) {
    if (out_hls[i] != out_ref[i]) pass = false;
  }
  if (!pass) return EXIT_FAILURE;
}
