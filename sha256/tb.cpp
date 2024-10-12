#include <cassert>
#include <fstream>
#include <vector>
#include <random>
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include "kernel.hpp"

static uint32_t ROTR(uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}
static uint32_t SHR(uint32_t x, int n) {
  return x >> n;
}
static uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}
static uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}
static uint32_t SIGMA0(uint32_t x) {
  return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22);
}
static uint32_t SIGMA1(uint32_t x) {
  return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25);
}
static uint32_t sigma0(uint32_t x) {
  return ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3);
}
static uint32_t sigma1(uint32_t x) {
  return ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10);
}

const uint32_t K[] = {
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

const uint32_t H0[] = {
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// Preprocess
void padding(std::vector<uint8_t>& in) {
  uint64_t bits = in.size() * 8;
  int num_pad = 56 - in.size() % 64;
  if (num_pad <= 0)  num_pad += 64;
  in.push_back(0x80);
  for (int i = 0; i < num_pad - 1; i++) in.push_back(0);
  for (int i = 7; i >= 0; i--) in.push_back((bits >> (i*8)) & 0xff);
}

// Swap byte order
std::vector<uint32_t> swap(const std::vector<uint8_t>& in) {
  std::vector<uint32_t> out;
  const uint8_t* ptr = in.data();
  for (int i = 0; i < in.size() / 4; i++) {
    uint32_t v = 0;
    for (int j = 0; j < 4; j++) v = (v << 8) | *(ptr++);
    out.push_back(v);
  }
  return out;
}

std::vector<uint32_t> sha256(const std::vector<uint32_t>& in) {
  // Initialize hash
  std::vector<uint32_t> H(8);
  for (int i = 0; i < 8; i++) H[i] = H0[i];

  const int num_blocks = in.size() / 16;
  for (int i = 0; i < num_blocks; i++) {
    // Message schedule
    uint32_t W[64];
    for (int j = 0; j < 64; j++) {
      if (j < 16) {
        W[j] = in[i * 16 + j];
      } else {
        W[j] = sigma1(W[j - 2]) + W[j - 7] + sigma0(W[j - 15]) + W[j - 16];
      }
    }

    // Hash
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (int j=0; j<64; j++) {
      uint32_t T1 = h + SIGMA1(e) + Ch(e, f, g) + K[j] + W[j];
      uint32_t T2 = SIGMA0(a) + Maj(a, b, c);
      h = g;
      g = f;
      f = e;
      e = d + T1;
      d = c;
      c = b;
      b = a;
      a = T1 + T2;
    }

    H[0] += a;
    H[1] += b;
    H[2] += c;
    H[3] += d;
    H[4] += e;
    H[5] += f;
    H[6] += g;
    H[7] += h;
  }

  return H;
}

int main(int argc, char** argv)
{
  std::vector<uint8_t> in_bytes;

//  {
//    std::ifstream is(argv[1], std::ios_base::binary);
//    assert(is);
//    char c;
//    while (is.get(c)) {
//      in_bytes.push_back(c);
//    }
//  }

  // Randomize input vector
  int num_bytes = 512 * 64;
  auto seed = []() { return 1; };
  std::mt19937 mt(seed());
  for (int i = 0; i < num_bytes; i++) {
    in_bytes.push_back(mt());
  }

  // Preprocess
  padding(in_bytes);
  auto in_words = swap(in_bytes);

  // Reference
  std::vector<uint32_t> ref = sha256(in_words);

  // Kernel
  block_t in_blocks[1024];
  hash_t out;
  int num_blocks = in_words.size() / 16;
  for (int i = 0; i < in_words.size(); i++) {
    in_blocks[i / 16][i % 16] = in_words[i];
  }
  kernel(in_blocks, num_blocks, &out);

  for (int i=0; i<8; i++) printf("%08x ", ref[i]);
  printf("\n");

  for (int i=0; i<8; i++) printf("%08x ", out[i]);
  printf("\n");

  // Check
  bool pass = true;
  for (int i=0; i<8; i++) if (out[i] != ref[i]) pass = false;
  if (!pass) return EXIT_FAILURE;
}

