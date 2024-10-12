#include <cassert>
#include <vector>
#include <random>

#include "kernel.hpp"

int fibonacci_recursive(const int n) {
  if (n == 0 || n == 1) {
    return n;
  } else {
    return fibonacci_recursive(n - 2) + fibonacci_recursive(n - 1);
  }
}

int main(int argc, char** argv)
{
  int out;
  kernel(&out);

  // Check
  bool pass = true;
  int expected = fibonacci_recursive(NUM);
  printf("expected:%d actual:%d\n", expected, out);
  if (out != expected) pass = false;
  if (!pass) return EXIT_FAILURE;
}
