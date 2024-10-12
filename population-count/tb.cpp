#include <cstdlib>

#include "kernel.hpp"

int main(int argc, char** argv)
{
  uint32_t test[] = { 0x00000000, 0xffffffff, 0xaaaaaaaa, 0x55555555 };
  int16_t expect[] = { 0, 32, 16, 16 };
  int16_t actual;

  for (int i = 0; i < 4; i++) {
    kernel(test[i], &actual);

    if (actual != expect[i]) {
      exit(EXIT_FAILURE);
    }
  }
}
