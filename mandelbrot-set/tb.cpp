#include <cassert>
#include <vector>
#include <complex>

#include "kernel.hpp"

void mandelbrot(
  const int width,
  const int height,
  const int max_iter,
  const float start_x,
  const float start_y,
  const float step_x,
  const float step_y,
  char* output
) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      std::complex<float> c(start_x + step_x * x, start_y + step_y * y);
      std::complex<float> z(0, 0);
      bool plot = true;
      for (int i = 0; i < max_iter; i++) {
        if (std::abs(z) > 2.0f) {
          plot = false;
          break;
        }
        z = z * z + c;
      }
      output[y * width + x] = plot ? '*' : ' ';
    }
  }
}

int main(int argc, char** argv)
{
  const int width = 32;
  const int height = 16;
  const int max_iter = 16;
  const float start_x = -2.0f;
  const float start_y = -1.0f;
  const float step_x = 2.5f / width;
  const float step_y = 2.0f / height;

  std::vector<char> golden(width * height);
  mandelbrot(width, height, max_iter, start_x, start_y, step_x, step_y, golden.data());

  char output[MAX_OUTPUT_SIZE];
  kernel(width, height, max_iter, start_x, start_y, step_x, step_y, output);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      printf("%c", output[y * width + x]);
    }
    printf("\n");
  }

  // Check
  bool pass = true;
  for (int i = 0; i < width * height; i++) {
    if (output[i] != golden[i]) pass = false;
  }
  if (!pass) return EXIT_FAILURE;
}
