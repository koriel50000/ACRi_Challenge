# 問題
ハフマン符号はデータ中によく出現する文字に対して短いビット列を割り当てることでデータを圧縮します。

与えられた符号表（各文字に対応する符号が格納されたテーブル）を使って、入力データを圧縮してください。符号は以下のように出力配列`out`の上位ビットから詰めて出力してください。

[[[bit-field
{ "reg": [
    { "name": "c2[5:3]",   "bits": 3, "type": 4 },
    { "name": "c1[2:0]",   "bits": 3, "type": 3 },
    { "name": "c0[1:0]",   "bits": 2, "type": 2 },
    { "name": "c3[7:3]",   "bits": 5, "type": 5 },
    { "name": "c2[2:0]",   "bits": 3, "type": 4 },
    { "name": "c5[12:11]", "bits": 2, "type": 7 },
    { "name": "c4[2:0]",   "bits": 3, "type": 6 },
    { "name": "c3[2:0]",   "bits": 3, "type": 5 },
    { "name": "c5[11:4]",  "bits": 8, "type": 7 },
    { "bits": 4 },
    { "name": "c5[3:0]",   "bits": 4, "type": 7 }
], "options": {
    "hspace": 400,
    "lanes": 5,
    "hflip": true,
    "bits": 40,
    "compact": true,
    "label": { "left": ["out[0]", "out[1]", "out[2]", "out[3]", "out[4]"] }
}}
]]]

## 引数

* data
  * 入力データ。
* code
  * 文字`i`の符号が`code[i]`に下詰めで格納されています。
* code_size
  * 文字`i`の符号のビット数が`code_size[i]`に格納されています。
* out
  * 符号を詰めて出力する配列。

## 参考

- [Wikipedia: ハフマン符号](https://ja.wikipedia.org/wiki/%E3%83%8F%E3%83%95%E3%83%9E%E3%83%B3%E7%AC%A6%E5%8F%B7)