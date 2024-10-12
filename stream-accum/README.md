# 問題

ストリームで入力される浮動小数点数をすべて足し合わせ、結果を出力してください。ただし、0.001以下の誤差は許容します。

`stream_end`の入力が`false`のとき、対応する`stream_data`の入力が有効です。`stream_end`の入力が`true`になったら結果を出力し終了してください。

## 参考

- [HLS ストリーム ライブラリ](https://japan.xilinx.com/html_docs/xilinx2020_2/vitis_doc/hls_stream_library.html)
- [Stream-based Interface](https://xilinx.github.io/Vitis_Libraries/utils/2020.2/guide/stream_based.html)