#include "kernel.hpp"
#include "image.hpp"
#include "params.hpp"

void threshold_padding_zero(hls::stream<uint64_t>& ins, int i) {
    if (i % 7 == 6) {
        for (int j = 0; j < 7; j++) {
            ins.write(0);
        }
    }
}

void input_stream(hls::stream<uint64_t>& ins) {
	// Conv_head

    // torch.Size([1, 3, 160, 160])
    for (int i = 0; i < 160 * 160; i++) {
        ins.write(images[i]);
    }

    // torch.Size([16, 3, 3, 3])
    for (int i = 0; i < 16 * 9; i++) {
        ins.write(backbone_model0_conv1_weight[i]);
    }
    // torch.Size([16, 7])
    for (int i = 0; i < 16 * 7; i++) {
        ins.write(backbone_model0_relu1_threshold[i]);
        threshold_padding_zero(ins, i);
    }

	// Conv_head ConvDPUnit

    // torch.Size([16, 1, 1, 16])
    for (int i = 0; i < 16 * 1; i++) {
        ins.write(backbone_model0_conv2_conv1_weight[i]);
    }
    // torch.Size([16, 14])
    for (int i = 0; i < 16 * 14; i++) {
        ins.write(backbone_model0_conv2_quant1_threshold[i]);
    }

    // torch.Size([16, 3, 3, 1])
    for (int i = 0; i < 16 * 9; i++) {
        ins.write(backbone_model0_conv2_conv2_weight[i]);
    }
    // torch.Size([16, 7])
    for (int i = 0; i < 16 * 7; i++) {
        ins.write(backbone_model0_conv2_relu2_threshold[i]);
        threshold_padding_zero(ins, i);
    }

    // torch.Size([16, 1, 1, 16])
    for (int i = 0; i < 16 * 1; i++) {
        ins.write(backbone_model1_conv1_conv1_weight[i]);
    }
    // torch.Size([16, 14])
    for (int i = 0; i < 16 * 14; i++) {
        ins.write(backbone_model1_conv1_quant1_threshold[i]);
    }

};

int main(int argc, char** argv)
{
    hls::stream<uint64_t> ins;
	int out[16];

    input_stream(ins);
	kernel(ins, out);
    printf("out[0]=%d\n", out[0]);

	return EXIT_FAILURE; // EXIT_SUCCESS;
}
