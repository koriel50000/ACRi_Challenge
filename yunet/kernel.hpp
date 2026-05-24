#pragma once
#include <stdint.h>
#include <ap_axi_sdata.h>
#include "hls_stream.h"

// @see ug1399, HLS Programmers Guide > Customizing-AXI4-Stream-Interfaces
using axis_data = ap_axis<64, 0, 0, 0, (AXIS_ENABLE_DATA | AXIS_ENABLE_LAST), true>;

extern "C" {
void kernel(
	hls::stream<axis_data>& ins,
	hls::stream<axis_data>& outs
);
}
