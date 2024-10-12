#include <algorithm>
#include <vector>
#include <random>
#include <cassert>
#include <cstring>

#include "kernel.hpp"

int main(int argc, char** argv)
{
  // Randomize input vector
  std::default_random_engine engine(1);
  std::uniform_int_distribution<int> dist(0, 25);

  std::string in_text(TEXT_SIZE, ' '), in_pattern(PATTERN_SIZE, ' ');
  std::vector<int> out(MAX_MATCH), ref(MAX_MATCH);

  auto copy_string = [](const std::string& str, std::string& text, int pos = 0) {
    assert(pos + str.size() <= text.size());
    std::copy(str.begin(), str.end(), text.begin() + pos);
  };

  auto test = [&]() {
    // Reference
    int ref_count = 0;
    for (int i = 0; i < TEXT_SIZE; i++) {
      bool match = true;
      for (int j = 0; j < PATTERN_SIZE; j++) {
        if (in_pattern[j] != '?' && (i + j >= TEXT_SIZE || in_text[i + j] != in_pattern[j])) {
          match = false;
        }
      }
      if (match) {
        ref[ref_count++] = i;
        if (ref_count >= MAX_MATCH) break;
      }
    }

    int out_count = 0;
    kernel(in_text.data(), in_pattern.data(), out.data(), &out_count);

    // Check
    bool pass = true;
    if (out_count != ref_count) pass = false;
    for (int i = 0; i < ref_count; i++) {
      if (out[i] != ref[i]) pass = false;
    }
    return pass;
  };

  // Pattern
  std::fill(in_pattern.begin(), in_pattern.end(), '?');
  copy_string("hoge", in_pattern);

  // Text
  std::generate(in_text.begin(), in_text.end(), [&]() { return 'a' + dist(engine); });
  copy_string("hoge", in_text, 0);
  copy_string("hoge", in_text, 4);
  copy_string("hoge", in_text, 1000);
  copy_string("hoge", in_text, TEXT_SIZE - 4);

  if (!test()) return EXIT_FAILURE;
}
