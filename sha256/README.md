# 問題

SHA256 アルゴリズムによるハッシュ関数を実装してください。前処理（パディング）はテストベンチで行い、カーネルではハッシュ計算のみを行います。

## カーネル引数

- `const block_t input[1024]`
  - メッセージブロックの配列
  - `block_t` は `hls::vector<uint32_t, 16>` で定義されます
  - 32bit 整数が 16 個で 512bit のメッセージブロックを表現します
- `const int size`
  - メッセージブロックの数
- `hash_t* output`
  - 求めたハッシュ値を出力するポインタ
  - `hash_t` は `hls::vector<uint32_t, 8>` で定義されます
  - 32bit 整数が 8 個で 256bit のハッシュ値を表現します

## hls::vector クラスについて

複数の要素を束ねて扱うのに便利なクラスです。512bit のメッセージブロックを表現するために `ap_uint<512>` 型を使うことも考えられますが、SHA256 の計算は 32bit 単位で行うため、32bit 単位でデータの出し入れができる `hls::vector` クラスを選択しました。

## 参考

- [SHA-2](https://ja.wikipedia.org/wiki/SHA-2)
- [Secure Hash Standard](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf)
- [SIMD 演算用の HLS ベクター型](https://japan.xilinx.com/html_docs/xilinx2020_2/vitis_doc/vitis_hls_coding_styles.html?hl=vector#hjd1600374477961__section_b4d_12v_1nb)