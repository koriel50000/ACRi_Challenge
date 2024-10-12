#include <complex>

#include "kernel.hpp"

uint32_t ROTR(const uint32_t x, const int n) {
	return (x >> n) | (x << (32 - n));
}

uint32_t SHR(const uint32_t x, const int n) {
	return x >> n;
}

uint32_t sigma0(const uint32_t x) {
	return ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3);
}

uint32_t sigma1(const uint32_t x) {
	return ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10);
}

uint32_t SIGMA1(const uint32_t x) {
	return ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25);
}

uint32_t Ch(const uint32_t x, const uint32_t y, const uint32_t z) {
	return (x & y) ^ (~x & z);
}

uint32_t SIGMA0(const uint32_t x) {
	return ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22);
}

uint32_t Maj(const uint32_t x, const uint32_t y, const uint32_t z) {
	return (x & y) ^ (x & z) ^ (y & z);
}

const uint32_t K[] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
	0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
	0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
	0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
	0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
	0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

void rotate_message(block_t& W, const uint32_t w) {
	for (int i = 0; i < 15; i++) {
#pragma HLS unroll
		W[i] = W[i + 1];
	}
	W[15] = w;
}

void sha256(block_t& W, hash_t& H) {
	hash_t A = H;

	int p = 0;
	for (int j = 0; j < 16; j++) {
		uint32_t w = W[0] + sigma0(W[1]) + W[9] + sigma1(W[14]);
		uint32_t t2 = SIGMA0(A[0]) + Maj(A[0], A[1], A[2]);
		uint32_t t1 = K[p++] + W[0] + A[7]
			+ SIGMA1(A[4]) + Ch(A[4], A[5], A[6]);

		A[7] = A[3] + t1;
		A[3] = t1 + t2;

		rotate_message(W, w);

		w = W[0] + sigma0(W[1]) + W[9] + sigma1(W[14]);
		t2 = SIGMA0(A[3]) + Maj(A[3], A[0], A[1]);
		t1 = K[p++] + W[0] + A[6]
			+ SIGMA1(A[7]) + Ch(A[7], A[4], A[5]);

		A[6] = A[2] + t1;
		A[2] = t1 + t2;

		rotate_message(W, w);

		w = W[0] + sigma0(W[1]) + W[9] + sigma1(W[14]);
		t2 = SIGMA0(A[2]) + Maj(A[2], A[3], A[0]);
		t1 = K[p++] + W[0] + A[5]
			+ SIGMA1(A[6]) + Ch(A[6], A[7], A[4]);

		A[5] = A[1] + t1;
		A[1] = t1 + t2;

		rotate_message(W, w);

		w = W[0] + sigma0(W[1]) + W[9] + sigma1(W[14]);
		t2 = SIGMA0(A[1]) + Maj(A[1], A[2], A[3]);
		t1 = K[p++] + W[0] + A[4]
			+ SIGMA1(A[5]) + Ch(A[5], A[6], A[7]);

		A[4] = A[0] + t1;
		A[0] = t1 + t2;

		rotate_message(W, w);
	}

	H += A;
}

void kernel(const block_t input[1024], const int size, hash_t* output) {
	static hash_t H = {
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
		0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
	};

	for (int i = 0; i < size; i++) {
		block_t W;
		for (int j = 0; j < 16; j++) {
			W[j] = input[i][j];
		}
		sha256(W, H);
	}

	*output = H;
}
