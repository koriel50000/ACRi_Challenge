# 問題

グレー画像の1ライン分、1024個の画素が入力されます。面積平均法を使って、指定されたサイズに縮小してください。

## 入力

* in
  * 入力ライン（1024画素）
* out_size
  * 出力ラインの画素数（64以上、1024未満とします）

## 出力

* out
  * 出力ライン

面積平均法は、入力ラインを水平方向に out_size 倍に拡大し、それから in_size 倍に縮小すると考えると整数精度の演算で実現できます（`in_size * out_size / in_size = out_size` となることから期待通り縮小できることが分かります）。

以下は入力が8画素、出力が3画素の例を示します。

[[[bit-field
{ "reg": [
    { "name": "in[7]", "bits": 3, "type": 3 },
    { "name": "in[6]", "bits": 3, "type": 2 },
    { "name": "in[5]", "bits": 3, "type": 7 },
    { "name": "in[4]", "bits": 3, "type": 6 },
    { "name": "in[3]", "bits": 3, "type": 5 },
    { "name": "in[2]", "bits": 3, "type": 4 },
    { "name": "in[1]", "bits": 3, "type": 3 },
    { "name": "in[0]", "bits": 3, "type": 2 },
    { "name": "out[2]", "bits": 8, "type": 4 },
    { "name": "out[1]", "bits": 8, "type": 3 },
    { "name": "out[0]", "bits": 8, "type": 2 }
], "options": {
    "hspace": 500,
    "lanes": 2,
    "hflip": true,
    "bits": 48,
    "compact": true
}}
]]]

`out[0]` は `(in[0] * 3 + in[1] * 3 + in[2] * 2 + 4) / 8`、`out[1]` は `(in[2] * 1 + in[3] * 3 + in[4] * 3 + in[5] * 1 + 4) / 8`、`out[2]` は `(in[5] * 2 + in[6] * 3 + in[7] * 3 + 4) / 8` により求められます。割る数の半分、`4` を足すことで四捨五入を行います。
