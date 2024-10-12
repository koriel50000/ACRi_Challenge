#include <algorithm>
#include <random>

#include "kernel.hpp"
#include "state.hpp"
#include "feature.hpp"

#include "state.cpp"
#include "feature.cpp"

int main(int argc, char** argv)
{
  // Randomize input vector
  std::default_random_engine engine(1);
  std::uniform_real_distribution<> dist(-1000, 1000);

  float in[SIZE];
  for (int i = 0; i < SIZE; i++) {
    in[i] = dist(engine);
  }

  bool reset = true;
  ap_uint<6> current_move = 0;
  float state_buffer[ROWS * COLUMNS * CHANNELS];

  kernel(reset, current_move, state_buffer);

  // Check
  bool pass = true;
  for (int i = 0; i < ROWS * COLUMNS * CHANNELS; i++) {
    if (out_hls[i] != in[i]) pass = false;
  }
  if (!pass) return EXIT_FAILURE;
}
