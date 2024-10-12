#include <algorithm>
#include <memory>
#include <vector>
#include <random>

#include "kernel.hpp"

struct node {
  using ptr_t = std::shared_ptr<node>;
  uint64_t ch, freq, code, code_size;
  ptr_t left, right;
  node(uint8_t c): ch(c), freq(0) {}
  node(ptr_t& l, ptr_t& r): freq(l->freq + r->freq), left(l), right(r) {}
  bool is_leaf() { return !left; }
};

void assign_code(node::ptr_t& n, uint64_t code = 0, uint64_t code_size = 0) {
  if (n->left) {
    assign_code(n->left , (code << 1) | 0, code_size + 1);
    assign_code(n->right, (code << 1) | 1, code_size + 1);
  } else {
    n->code = code; n->code_size = code_size;
  }
};

auto make_huffman(const auto& nodes) {
  auto tree = nodes;
  while (tree.size() > 1) {
    std::sort(tree.begin(), tree.end(), [](auto& a, auto& b) { return a->freq > b->freq; });
    auto a = tree.back(); tree.pop_back();
    auto b = tree.back(); tree.pop_back();
    tree.emplace_back(new node(a, b));
  }
  assign_code(tree[0]);
  return tree[0];
}

void encode(uint8_t data[SIZE], uint64_t code[256], uint64_t code_size[256], uint8_t out[SIZE*8]) {
  uint64_t out_bits = 0;
  for (int i = 0; i < SIZE; i++) {
    uint64_t c = code     [data[i]];
    uint64_t s = code_size[data[i]];
    while (s > 0) {
      int b = std::min(s, 8 - out_bits % 8);
      out[out_bits / 8] |= ((c >> (s - b)) & ((1 << b) - 1)) << (8 - out_bits % 8 - b);
      out_bits += b;
      s -= b;
    }
  }
}

void decode(uint8_t in[SIZE*8], const node::ptr_t& root, uint8_t out[SIZE]) {
  int in_index = 0;
  int in_code = 0;
  int in_bits = 0;
  int dec_index = 0;
  node::ptr_t n = root;
  while (dec_index < SIZE) {
    if (in_bits == 0) {
      in_code = in[in_index++];
      in_bits = 8;
    }
    int bit = (in_code >> 7) & 1;
    in_code = in_code << 1;
    in_bits--;
    n = bit == 0 ? n->left : n->right;
    if (n->is_leaf()) {
      out[dec_index++] = n->ch;
      n = root;
    }
  }
}

int main(int argc, char** argv)
{
  // Randomize input vector
  auto seed = []() { return 1; };
  std::default_random_engine engine(seed());
  std::normal_distribution<> dist(0, 40);
  auto rand = [&]() { return std::min(std::abs(int(dist(engine))), 255); };

  std::vector<node::ptr_t> nodes;
  for (int i = 0; i < 256; i++) {
    nodes.emplace_back(new node(i));
  }

  uint8_t data[SIZE];
  for (int i = 0; i < SIZE; i++) {
    data[i] = rand();
    nodes[data[i]]->freq++;
  }

  auto root = make_huffman(nodes);

  uint64_t code[256], code_size[256];;
  for (int i = 0; i < nodes.size(); i++) {
    code[i] = nodes[i]->code;
    code_size[i] = nodes[i]->code_size;
    //printf("char: %3d, freq: %4ld, code: %012lx, size: %2ld\n", i, nodes[i]->freq, code[i], code_size[i]);
  }

  uint8_t out[SIZE*8];

  //encode(data, code, code_size, out);
  kernel(data, code, code_size, out);

  uint8_t dec[SIZE];
  decode(out, root, dec);

  // Check
  bool pass = true;
  for (int i = 0; i < SIZE; i++) {
    //printf("%02x, %02x\n", data[i], dec[i]);
    if (data[i] != dec[i]) pass = false;
  }
  printf("%s\n", pass ? "Pass" : "Fail");
  if (!pass) return EXIT_FAILURE;
}
