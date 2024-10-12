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

このチャレンジでは CNN 全体をひとつのカーネル内に実装します。

# 問題
入力される 28x28 白黒画像に対して CNN による推論を行い、最後の全結合層の出力を出力してください。

## 入力
- `in[28 * 28 * 1]`
  - 高さ 28、幅 28、1bit 白黒画像
  - HWC フォーマット
    - 座標 `(x, y)` の画素が `in[y * 28 + x]` に格納されています
- `conv0_weight[16 * 5 * 5 * 1]`
  - 5x5 符号付き 2bit 畳み込みフィルタ
  - 出力チャネル数 16
  - 符号付き 2bit
- `threshold0[3]`
  - ビット数削減のための閾値
- `conv1_weight[16 * 5 * 5 * 16]`
  - 5x5 符号付き 2bit 畳み込みフィルタ
  - 入力チャネル数 16
  - 出力チャネル数 16
  - 符号付き 2bit
- `threshold1[3]`
  - ビット数削減のための閾値
- `matmul0_weight[256 * 10]`
  - 入力チャネル数 256
  - 出力チャネル数 10
  - 符号付き 2bit

## 出力
- `out[10]`
  - 10 チャネル
