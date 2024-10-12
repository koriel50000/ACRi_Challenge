#include <cassert>
#include <vector>
#include <random>

#include "kernel.hpp"

template<typename T>
void vector_to_stream(
  const std::vector<T>& vec,
  hls::stream<T>& stream_data,
  hls::stream<bool>& stream_end
) {
  for (const auto& x: vec) {
    stream_data.write(x);
    stream_end.write(false);
  }
  stream_end.write(true);
}

int main(int argc, char** argv)
{
  // Randomize input vector
  //std::random_device seed;
  auto seed = []() { return 1; };
  std::default_random_engine engine(seed());
  std::uniform_real_distribution<float> dist(-1000.0, 1000.0);

  const int size = 256;
  std::vector<float> in(size);
  float accum = 0;
  for (int i=0; i<size; i++) {
    in[i] = dist(engine);
    accum += in[i];
  }

  hls::stream<float> stream_data;
  hls::stream<bool> stream_end;
  vector_to_stream(in, stream_data, stream_end);

  float output;
  kernel(stream_data, stream_end, &output);

  // Check
  bool pass = true;
  if (!(std::abs(accum - output) <= 1e-3)) pass = false;
  if (!stream_data.empty()) pass = false;
  if (!stream_end.empty()) pass = false;
  if (!pass) return EXIT_FAILURE;
}
