#include "kernel.hpp"

// @see https://ja.wikipedia.org/wiki/%E3%83%A2%E3%83%B3%E3%82%B4%E3%83%A1%E3%83%AA%E4%B9%97%E7%AE%97
// @see https://tex2e.github.io/blog/crypto/montgomery-mul
// モンゴメリ乗算

typedef ap_uint<256> word_t;
typedef ap_uint<512> dword_t;
typedef ap_uint<128> half_t;
typedef ap_uint<257> c_word_t;
typedef ap_uint<514> c_dword_t;

const int Rind = 256;
const c_word_t R = c_word_t(1) << Rind;
// R > N かつ Rの約数は2のみで N=p*q (p,q:素数) だから gcd(N,R)=1

// @see Rが2の冪の時のN'の効率的な求め方
word_t init_ninv(const word_t n) {
	word_t t = 0;
	word_t ninv = 0;
	for (int i = 0; i < Rind; i++) {
		if (t[0] == 0) {
			t += n;
			ninv[i] = 1;
		}
		t >>= 1;
	}
	return ninv;
}

// @thanks https://acri-vhls-challenge.web.app/user/basaro/code/BXeyDJIdcMVmDPoqESFx
word_t init_r2(const word_t n) {
	// R2 = R * R mod N
	static ap_uint<768> r2 = R * R; // r[512] = 1
	// 筆算で割り算の余りを計算
	for (int i = Rind * 2; i >= 0; --i) {
		ap_uint<768> div = ((ap_uint<768>)n) << i;
		if (r2 > div) {
			r2 -= div;
		}
	}
	return r2;
}

// @thanks https://acri-vhls-challenge.web.app/user/srmfsan/code/4YkOSTtmltFSsDZ6clXh
// MontgomeryReduction
word_t MR(const c_dword_t x, const word_t n, const word_t ninv) {
	// t = (T + ((T * N') mod R) * N) / R
	word_t x_ninv = (x * ninv).range(Rind - 1, 0);
	dword_t x_ninv_n = x_ninv * n;
	c_dword_t x_x_ninv_n = x + x_ninv_n; // 桁上がりを考慮
	c_word_t t = x_x_ninv_n >> Rind;
	// if t >= N then return t - N else return t
	if (t >= (c_word_t)n) {
	       return t - (c_word_t)n; // FIXME 桁上がりは？
	} else {
	       return t;
	}
}

// MontgomeryModularMultiplication
//word_t MMM(const word_t a, const half_t b, const word_t n, const word_t ninv, const word_t r2) {
word_t MMM(const word_t a, const word_t b, const word_t n, const word_t ninv, const word_t r2) {
	// c = MR(MR(a * b) * R2)
	// R2 = R * R mod N
	word_t ab = MR(a * b, n, ninv);
	return MR(ab * r2, n, ninv);
}

int count_leading_zeros(const half_t bits) {
        int i;
        for (i = 127; i >= 0; --i) {
                if (bits[i] == 1) break;
        }
        return i;
}

void kernel(const half_t plain, const half_t e, const word_t n, word_t* encrypted) {
	// *encrypted = (plain ^ e) % n
	// @see https://ja.wikipedia.org/wiki/%E5%86%AA%E5%89%B0%E4%BD%99
	// 冪剰余 > さらなる最適化
	// FIXME 最上位ビットまで使っていそうなのでメリットなし
	// p = gen_random(bits, gen) | (mp::cpp_int(1) << (bits - 1)) | 1;

	word_t ninv = init_ninv(n);
	word_t r2 = init_r2(n);

	word_t result = 1;
	//word_t m = plain % n;
	//int count = count_leading_zeros(e);

	//for (int i = count; i >= 0; --i) {
	//	result = MMM(result, n, ninv, r2);
	//	if (e[i] == 1) {
	//		result = MMM(result, m, n, ninv, r2);
	//	}
	//}
	word_t b = plain;
	for (int i = 0; i < 128; i++) {
		if (e[i] == 1) {
			result = MMM(result, b, n, ninv, r2);
		}
		b = MMM(b, b, n, ninv, r2);
	}

	*encrypted = result;
}
