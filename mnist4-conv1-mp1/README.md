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

このチャレンジでは CNN のConv2D 層と Max Pooling 層（conv1 と maxpool1）をひとつのカーネル内に実装します。

# 問題
入力される 12x12 特徴マップに対してカーネルサイズ 5x5 の畳み込み演算とビット削減を行い、さらに 2x2 の Max Pooling 処理を行って、結果を出力してください。

畳み込み演算、ビット削減、Max Pooling 処理の実装については他のチャレンジの説明を参考にしてください。

## 入力
- `in[12 * 12 * 16]`
  - 高さ 12
  - 幅 12
  - 16 チャネル
  - 符号なし 2bit 整数
  - HWC フォーマット
    - `座標 `(x, y)`、`c` チャネルの値が `in[(y * 12 + x) * 16 + c]` に格納されています
- `weight[16 * 5 * 5 * 16]`
  - 5x5 符号付き 2bit 畳み込みフィルタ
  - 入力チャネル数 16
  - 出力チャネル数 16
  - 符号付き 2bit
- `threshold[3]`
  - ビット数削減のための閾値

## 出力
- `out[8 * 8 * 16]`
  - 高さ 8
  - 幅 8
  - 16 チャネル
  - 符号なし 2bit 整数