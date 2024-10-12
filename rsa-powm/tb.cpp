#include <cassert>
#include <vector>
#include <random>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/miller_rabin.hpp>

#include "kernel.hpp"

namespace mp = boost::multiprecision;

mp::cpp_int gcd(const mp::cpp_int& a, const mp::cpp_int& b) {
  auto a_mod_b = a % b;
  return a_mod_b == 0 ? b : gcd(b, a_mod_b);
}

mp::cpp_int lcm(const mp::cpp_int& a, const mp::cpp_int& b) {
  return a * b / gcd(a, b);
}

template<typename Engine>
mp::cpp_int gen_random(unsigned bits, Engine& gen) {
  std::uniform_int_distribution<> dist(0, 1);
  mp::cpp_int r = 0;
  for (unsigned i = 0; i < bits; i++) r |= mp::cpp_int(dist(gen)) << i;
  return r;
}

template<typename Engine>
mp::cpp_int gen_prime(unsigned bits, Engine& gen) {
  while (true) {
    mp::cpp_int p = gen_random(bits, gen) | (mp::cpp_int(1) << (bits - 1)) | 1;
    if (mp::miller_rabin_test(p, 25, gen)) return p;
  }
}

template<int N>
ap_uint<N> to_ap_uint(const mp::cpp_int& a) {
  ap_uint<N> b;
  for (int i = 0; i < N; i++) b[i] = static_cast<int>((a >> i) & 1);
  return b;
}

int main(int argc, char** argv)
{
  //std::random_device seed;
  auto seed = []() { return 2; };
  std::default_random_engine gen(seed());

  mp::cpp_int p = gen_prime(128, gen);
  mp::cpp_int q = gen_prime(128, gen);
  mp::cpp_int n = p * q;
  mp::cpp_int l = lcm(p - 1, q - 1);
  mp::cpp_int e, d;

  while (true) {
    // Generate public key
    e = gen_prime(128, gen);

    // Calculate private key
    mp::cpp_int x1 = 1, y1 = 0, z1 = l;
    mp::cpp_int x2 = 0, y2 = 1, z2 = e;
    while (z2 > 0) {
      mp::cpp_int a = z1 / z2;
      mp::cpp_int x3 = x1 - a * x2;
      mp::cpp_int y3 = y1 - a * y2;
      mp::cpp_int z3 = z1 - a * z2;
      x1 = x2; y1 = y2; z1 = z2;
      x2 = x3; y2 = y3; z2 = z3;
    }
    if (y1 < 0) continue;
    d = std::max(x1, y1);
    break;
  }

  std::cout << "p = " << p << std::endl;
  std::cout << "q = " << q << std::endl;
  std::cout << "n = " << n << std::endl;
  std::cout << "l = " << l << std::endl;
  std::cout << "e = " << e << std::endl;
  std::cout << "d = " << d << std::endl;

  // Generate random message
  mp::cpp_int plain = gen_random(128, gen);
  ap_uint<128> hls_plain = to_ap_uint<128>(plain);

  // Run kernel
  ap_uint<256> hls_encrypted;
  kernel(hls_plain, to_ap_uint<128>(e), to_ap_uint<256>(n), &hls_encrypted);

  // Golden
  mp::cpp_int encrypted = mp::powm(plain, e, n);
  mp::cpp_int decrypted = mp::powm(encrypted, d, n);

  std::cout << "plain              = " << plain         << std::endl;
  std::cout << "encrypted (hls)    = " << hls_encrypted << std::endl;
  std::cout << "encrypted (golden) = " << encrypted     << std::endl;
  std::cout << "decrypted          = " << decrypted     << std::endl;

  // Check
  bool pass = true;
  if (hls_encrypted != to_ap_uint<256>(encrypted)) pass = false;
  if (!pass) return EXIT_FAILURE;
}
