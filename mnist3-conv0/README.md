# はじめに
このチャレンジは DSF コラボ記念 HLS チャレンジの一部です。DSF コラボ記念 HLS チャレンジでは、複数のチャレンジを通して MNIST の手書き数字画像から数字を認識する以下の CNN（Convolutional Neural Network）を実装します。

```
              conv0               maxpool0
            ┌────────┐            ┌───────┐
   28x28x1  │ Conv2D │  24x24x16  │MaxPool│  12x12x16
  ─────────►│16x5x5x1├───────────►│  2x2  ├────────────┐
   (Input)  │ no pad │            │       │            │
            └────────┘            └───────┘            │
                                                       │
             ┌─────────────────────────────────────────┘
             │
             │       conv1             maxpool1
             │    ┌─────────┐          ┌───────┐
             │    │ Conv2D  │  8x8x16  │MaxPool│  4x4x16
             └───►│16x5x5x16├─────────►│  2x2  ├──────────┐
                  │ no pad  │          │       │          │
                  └─────────┘          └───────┘          │
                                                          │
                   ┌──────────────────────────────────────┘
                   │
                   │                     matmul0
                   │     ┌───────┐       ┌──────┐
                   │     │Flatten│  256  │MatMul│   10
                   └────►│       ├──────►│10x256├─────────►
                         │       │       │      │ (Output)
                         └───────┘       └──────┘
```

DSF コラボ記念 HLS チャレンジでは以下の 6 つのチャレンジを開催します。

- 初級：[mnist1-matmul0](https://acri-vhls-challenge.web.app/challenge/mnist1-matmul0) 全結合層 matmul0 を実装 
- 初級：[mnist2-maxpool0](https://acri-vhls-challenge.web.app/challenge/mnist2-maxpool0) Max Pooling 層 maxpool0 を実装 
- 中級：[mnist3-conv0](https://acri-vhls-challenge.web.app/challenge/mnist3-conv0) Conv2D 層 conv0 を実装 
- 中級：[mnist4-conv1-mp1](https://acri-vhls-challenge.web.app/challenge/mnist4-conv1-mp1) conv1 と maxpool1 をひとつのカーネルに実装
- 上級：[mnist5-all](https://acri-vhls-challenge.web.app/challenge/mnist5-all) CNN すべてをひとつのカーネルに実装
- 上級：[mnist6-free](https://acri-vhls-challenge.web.app/challenge/mnist6-free) 手書き数字認識を自由に実装 

このチャレンジでは CNN の最初の Conv2D 層（上図 conv0）を実装します。

# 問題
入力される 28x28 白黒画像に対してカーネルサイズ 5x5 の畳み込み演算を行い、さらにビット数を削減した結果を出力してください。畳み込みのフィルタとして符号付き 2bit の重みが与えられます。また、ビット数の削減に使用する閾値が与えられます。

## 入力
- `in[28 * 28 * 1]`
  - 高さ 28、幅 28、1bit 白黒画像
  - HWC フォーマット
    - 座標 `(x, y)` の画素が `in[y * 28 + x]` に格納されています
- `weight[16 * 5 * 5 * 1]`
  - 5x5 符号付き 2bit 畳み込みフィルタ
  - 出力チャネル数 16
  - 符号付き 2bit
- `threshold[3]`
  - ビット数削減のための閾値

## 出力
- `out[24 * 24 * 16]`
  - 高さ 24
  - 幅 24
  - 16 チャネル
  - 符号なし 2bit 整数

演算内容の詳細についてはテストベンチのリファレンス実装を参照してください。

## 畳み込み演算について
入力を $ x $、重みを $ w $ とするとき、出力 $ y $ を次のように求めます。

$$
y_{oy, ox, oc} = \sum_{ky}^{5} \sum_{kx}^{5} \sum_{ic} x_{oy + ky, ox + kx, ic} w_{oc, ky, kx, ic}
$$

パディングなしで畳み込みするため入力よりも出力の方が高さと幅が小さくなります。

## ビット数削減演算について
入力を $ x $、閾値を $ t_i $（$ i \in \lbrace 0, 1, 2 \rbrace $）とするとき、$ x \ge t_{i-1} $ を満たす最大の $ i $ を出力します。条件を満たす $ i $ が 存在しない場合は $ 0 $ を出力します。出力は符号なし 2bit 整数となります。
