# 問題
256 bit の RSA 暗号による暗号化を行ってください。

## 入力

* plain
  * 暗号化対象のメッセージ
* e
  * 秘密鍵
* n
  * 秘密のふたつの素数 p と q の積（n の素因数分解が難しいことが RSA 暗号の安全性を保証しています）

## 出力

* encrypted
  * 入力メッセージを a としたとき、以下の式で求めた暗号文 b

$$
\displaystyle b = a ^ e \mod n
$$

## 参考

* [Wikipedia: RSA暗号](https://ja.wikipedia.org/wiki/RSA%E6%9A%97%E5%8F%B7)