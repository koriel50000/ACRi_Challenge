#pragma once
#include <ap_axi_sdata.h>
#include "hls_stream.h"

// @see ug1399, HLS Programmers Guide > Customizing-AXI4-Stream-Interfaces
using axis_data64 = ap_axis<64, 0, 0, 0, (AXIS_ENABLE_DATA | AXIS_ENABLE_LAST), true>;
using axis_data8 = ap_axis<8, 0, 0, 0, (AXIS_ENABLE_DATA | AXIS_ENABLE_LAST), true>;

extern "C" {
void yunet(
	hls::stream<axis_data64>& ins,
	hls::stream<axis_data8>& outs
);
}
