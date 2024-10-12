#include <cstdlib>

#include "kernel.hpp"

int main(int argc, char** argv)
{
  int a = 1;
  int b = 2;
  int c;

  kernel(a, b, &c);

  if (c != a+b) {
    exit(EXIT_FAILURE);
  }
}