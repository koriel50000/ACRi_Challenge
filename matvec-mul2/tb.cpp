#include <algorithm>
#include <vector>
#include <random>

#include "kernel.hpp"

int main(int argc, char** argv)
{
  // Randomize input vector
  std::default_random_engine engine(1);
  std::uniform_real_distribution<float> dist(-10.0, 10.0);

  std::vector<float> in_mat(SIZE * SIZE), in_vec(SIZE), out(SIZE);
  std::vector<double> ref(SIZE);

  std::generate(in_mat.begin(), in_mat.end(), [&]() { return dist(engine); });
  std::generate(in_vec.begin(), in_vec.end(), [&]() { return dist(engine); });

  // Reference
  for (int i = 0; i < SIZE; i++) {
    double acc = 0;
    for (int j = 0; j < SIZE; j++) {
      acc += (double) in_mat[i * SIZE + j] * (double) in_vec[j];
    }
    ref[i] = acc;
  }

  kernel(in_mat.data(), in_vec.data(), out.data());

  // Check
  bool pass = true;
  for (int i = 0; i < SIZE; i++) {
    if (!(std::abs(out[i] - ref[i]) <= 1e-2)) pass = false;
  }
  if (!pass) return EXIT_FAILURE;
}
