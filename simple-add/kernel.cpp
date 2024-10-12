#include "kernel.hpp"

// @see https://docs.xilinx.com/r/ja-JP/ug1399-vitis-hls/pragma-HLS-interface
// とりあえず AIX-Steam を使っておけばよい？
void kernel(int a, int b, int* c) {
#pragma HLS interface axis port=a
#pragma HLS interface axis port=b
#pragma HLS interface axis port=c

	*c = a + b;
}
