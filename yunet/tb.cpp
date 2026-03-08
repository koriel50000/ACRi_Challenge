#include "kernel.hpp"
#include "image.hpp"
#include "params.hpp"

void input_stream(hls::stream<uint64_t>& ins);

int main(int argc, char** argv)
{
	hls::stream<uint64_t> ins;
	int out[16];

	input_stream(ins);
	kernel(ins, out);
	printf("out[0]=%d\n", out[0]);

	return EXIT_FAILURE; // EXIT_SUCCESS;
}

void threshold_padding_zero(hls::stream<uint64_t>& ins, int i) {
	if (i % 7 == 6) {
		for (int j = 0; j < 7; j++) {
			ins.write(0);
		}
	}
}

void input_stream(hls::stream<uint64_t>& ins) {
	// YuNetBackbone stage0
	// Conv_head

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

	// torch.Size([1, 3, 160, 160])
	for (int i = 0; i < 160 * 160; i++) {
		ins.write(images[i]);
	}

	// torch.Size([16, 1, 1, 9])
	for (int i = 0; i < 16 * 1; i++) {
		ins.write(backbone_model0_conv2_conv2_weight[i]);
	}
	// torch.Size([16, 7])
	for (int i = 0; i < 16 * 7; i++) {
		ins.write(backbone_model0_conv2_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNetBackbone stage1
	// YuNetBackbone Conv4layerBlock 1

	// torch.Size([16, 1, 1, 16])
	for (int i = 0; i < 16 * 1; i++) {
		ins.write(backbone_model1_conv1_conv1_weight[i]);
	}
	// torch.Size([16, 14])
	for (int i = 0; i < 16 * 14; i++) {
		ins.write(backbone_model1_conv1_quant1_threshold[i]);
	}

	// torch.Size([16, 1, 1, 9])
	for (int i = 0; i < 16 * 1; i++) {
		ins.write(backbone_model1_conv1_conv2_weight[i]);
	}
	// torch.Size([16, 7])
	for (int i = 0; i < 16 * 7; i++) {
		ins.write(backbone_model1_conv1_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNetBackbone Conv4layerBlock 2

	// torch.Size([64, 1, 1, 16])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(backbone_model1_conv2_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(backbone_model1_conv2_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(backbone_model1_conv2_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(backbone_model1_conv2_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNetBackbone stage2
	// YuNetBackbone Conv4layerBlock 1

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(backbone_model2_conv1_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(backbone_model2_conv1_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(backbone_model2_conv1_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(backbone_model2_conv1_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNetBackbone Conv4layerBlock 2

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(backbone_model2_conv2_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(backbone_model2_conv2_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(backbone_model2_conv2_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(backbone_model2_conv2_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNetBackbone stage3
	// YuNetBackbone Conv4layerBlock 1

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(backbone_model3_conv1_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(backbone_model3_conv1_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(backbone_model3_conv1_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(backbone_model3_conv1_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNetBackbone Conv4layerBlock 2

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(backbone_model3_conv2_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(backbone_model3_conv2_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(backbone_model3_conv2_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(backbone_model3_conv2_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNetBackbone stage4
	// YuNetBackbone Conv4layerBlock 1

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(backbone_model4_conv1_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(backbone_model4_conv1_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(backbone_model4_conv1_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(backbone_model4_conv1_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNetBackbone Conv4layerBlock 2

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(backbone_model4_conv2_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(backbone_model4_conv2_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(backbone_model4_conv2_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(backbone_model4_conv2_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNetBackbone stage5
	// YuNetBackbone Conv4layerBlock 1

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(backbone_model5_conv1_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(backbone_model5_conv1_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(backbone_model5_conv1_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(backbone_model5_conv1_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNetBackbone Conv4layerBlock 2

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(backbone_model5_conv2_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(backbone_model5_conv2_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(backbone_model5_conv2_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(backbone_model5_conv2_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// TFPN stride32
	// TFPN ConvDPUnit

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(neck_lateral_convs_2_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(neck_lateral_convs_2_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(neck_lateral_convs_2_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(neck_lateral_convs_2_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// TFPN stride16
	// TFPN ConvDPUnit

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(neck_lateral_convs_1_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(neck_lateral_convs_1_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(neck_lateral_convs_1_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(neck_lateral_convs_1_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// TFPN stride8
	// TFPN ConvDPUnit

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(neck_lateral_convs_0_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(neck_lateral_convs_0_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(neck_lateral_convs_0_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(neck_lateral_convs_0_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNet_Head stride8
	// YuNet_Head shared ConvDPUnit

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(bbox_head_multi_level_share_convs_0_0_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(bbox_head_multi_level_share_convs_0_0_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(bbox_head_multi_level_share_convs_0_0_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(bbox_head_multi_level_share_convs_0_0_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNet_Head stride16
	// YuNet_Head shared ConvDPUnit

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(bbox_head_multi_level_share_convs_1_0_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(bbox_head_multi_level_share_convs_1_0_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(bbox_head_multi_level_share_convs_1_0_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(bbox_head_multi_level_share_convs_1_0_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNet_Head stride32
	// YuNet_Head shared ConvDPUnit

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		ins.write(bbox_head_multi_level_share_convs_2_0_conv1_weight[i]);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 14; i++) {
		ins.write(bbox_head_multi_level_share_convs_2_0_quant1_threshold[i]);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		ins.write(bbox_head_multi_level_share_convs_2_0_conv2_weight[i]);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 7; i++) {
		ins.write(bbox_head_multi_level_share_convs_2_0_relu2_threshold[i]);
		threshold_padding_zero(ins, i);
	}

	// YuNet_Head cls ConvDPUnit
	// YuNet_Head stride8

	// torch.Size([1, 1, 1, 64])
	for (int i = 0; i < 1 * 1*4; i++) {
		ins.write(bbox_head_multi_level_cls_0_conv1_weight[i]);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 14; i++) {
		ins.write(bbox_head_multi_level_cls_0_quant1_threshold[i]);
	}

	// torch.Size([1, 1, 1, 9])
	for (int i = 0; i < 1 * 1; i++) {
		ins.write(bbox_head_multi_level_cls_0_conv2_weight[i]);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 14; i++) {
		ins.write(bbox_head_multi_level_cls_0_quant2_threshold[i]);
	}

	// YuNet_Head stride16

	// torch.Size([1, 1, 1, 64])
	for (int i = 0; i < 1 * 1*4; i++) {
		ins.write(bbox_head_multi_level_cls_1_conv1_weight[i]);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 14; i++) {
		ins.write(bbox_head_multi_level_cls_1_quant1_threshold[i]);
	}

	// torch.Size([1, 1, 1, 9])
	for (int i = 0; i < 1 * 1; i++) {
		ins.write(bbox_head_multi_level_cls_1_conv2_weight[i]);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 14; i++) {
		ins.write(bbox_head_multi_level_cls_1_quant2_threshold[i]);
	}

	// YuNet_Head stride32

	// torch.Size([1, 1, 1, 64])
	for (int i = 0; i < 1 * 1*4; i++) {
		ins.write(bbox_head_multi_level_cls_2_conv1_weight[i]);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 14; i++) {
		ins.write(bbox_head_multi_level_cls_2_quant1_threshold[i]);
	}

	// torch.Size([1, 1, 1, 9])
	for (int i = 0; i < 1 * 1; i++) {
		ins.write(bbox_head_multi_level_cls_2_conv2_weight[i]);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 14; i++) {
		ins.write(bbox_head_multi_level_cls_2_quant2_threshold[i]);
	}

	// YuNet_Head bbox ConvDPUnit
	// YuNet_Head stride8

	// torch.Size([4, 1, 1, 64])
	for (int i = 0; i < 4 * 1*4; i++) {
		ins.write(bbox_head_multi_level_bbox_0_conv1_weight[i]);
	}
	// torch.Size([4, 14])
	for (int i = 0; i < 4 * 14; i++) {
		ins.write(bbox_head_multi_level_bbox_0_quant1_threshold[i]);
	}

	// torch.Size([4, 1, 1, 9])
	for (int i = 0; i < 4 * 1; i++) {
		ins.write(bbox_head_multi_level_bbox_0_conv2_weight[i]);
	}
	// torch.Size([4, 14])
	for (int i = 0; i < 4 * 14; i++) {
		ins.write(bbox_head_multi_level_bbox_0_quant2_threshold[i]);
	}

	// YuNet_Head stride16

	// torch.Size([4, 1, 1, 64])
	for (int i = 0; i < 4 * 1*4; i++) {
		ins.write(bbox_head_multi_level_bbox_1_conv1_weight[i]);
	}
	// torch.Size([4, 14])
	for (int i = 0; i < 4 * 14; i++) {
		ins.write(bbox_head_multi_level_bbox_1_quant1_threshold[i]);
	}

	// torch.Size([4, 1, 1, 9])
	for (int i = 0; i < 4 * 1; i++) {
		ins.write(bbox_head_multi_level_bbox_1_conv2_weight[i]);
	}
	// torch.Size([4, 14])
	for (int i = 0; i < 4 * 14; i++) {
		ins.write(bbox_head_multi_level_bbox_1_quant2_threshold[i]);
	}

	// YuNet_Head stride32

	// torch.Size([4, 1, 1, 64])
	for (int i = 0; i < 4 * 1*4; i++) {
		ins.write(bbox_head_multi_level_bbox_2_conv1_weight[i]);
	}
	// torch.Size([4, 14])
	for (int i = 0; i < 4 * 14; i++) {
		ins.write(bbox_head_multi_level_bbox_2_quant1_threshold[i]);
	}

	// torch.Size([4, 1, 1, 9])
	for (int i = 0; i < 4 * 1; i++) {
		ins.write(bbox_head_multi_level_bbox_2_conv2_weight[i]);
	}
	// torch.Size([4, 14])
	for (int i = 0; i < 4 * 14; i++) {
		ins.write(bbox_head_multi_level_bbox_2_quant2_threshold[i]);
	}

	// YuNet_Head obj ConvDPUnit
	// YuNet_Head stride8

	// torch.Size([1, 1, 1, 64])
	for (int i = 0; i < 1 * 1*4; i++) {
		ins.write(bbox_head_multi_level_obj_0_conv1_weight[i]);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 14; i++) {
		ins.write(bbox_head_multi_level_obj_0_quant1_threshold[i]);
	}

	// torch.Size([1, 1, 1, 9])
	for (int i = 0; i < 1 * 1; i++) {
		ins.write(bbox_head_multi_level_obj_0_conv2_weight[i]);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 14; i++) {
		ins.write(bbox_head_multi_level_obj_0_quant2_threshold[i]);
	}

	// YuNet_Head stride16

	// torch.Size([1, 1, 1, 64])
	for (int i = 0; i < 1 * 1*4; i++) {
		ins.write(bbox_head_multi_level_obj_1_conv1_weight[i]);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 14; i++) {
		ins.write(bbox_head_multi_level_obj_1_quant1_threshold[i]);
	}

	// torch.Size([1, 1, 1, 9])
	for (int i = 0; i < 1 * 1; i++) {
		ins.write(bbox_head_multi_level_obj_1_conv2_weight[i]);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 14; i++) {
		ins.write(bbox_head_multi_level_obj_1_quant2_threshold[i]);
	}

	// YuNet_Head stride32

	// torch.Size([1, 1, 1, 64])
	for (int i = 0; i < 1 * 1*4; i++) {
		ins.write(bbox_head_multi_level_obj_2_conv1_weight[i]);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 14; i++) {
		ins.write(bbox_head_multi_level_obj_2_quant1_threshold[i]);
	}

	// torch.Size([1, 1, 1, 9])
	for (int i = 0; i < 1 * 1; i++) {
		ins.write(bbox_head_multi_level_obj_2_conv2_weight[i]);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 14; i++) {
		ins.write(bbox_head_multi_level_obj_2_quant2_threshold[i]);
	}

	// YuNet_Head kps ConvDPUnit
	// YuNet_Head stride8

	// torch.Size([10, 1, 1, 64])
	for (int i = 0; i < 10 * 1*4; i++) {
		ins.write(bbox_head_multi_level_kps_0_conv1_weight[i]);
	}
	// torch.Size([10, 14])
	for (int i = 0; i < 10 * 14; i++) {
		ins.write(bbox_head_multi_level_kps_0_quant1_threshold[i]);
	}

	// torch.Size([10, 1, 1, 9])
	for (int i = 0; i < 10 * 1; i++) {
		ins.write(bbox_head_multi_level_kps_0_conv2_weight[i]);
	}
	// torch.Size([10, 14])
	for (int i = 0; i < 10 * 14; i++) {
		ins.write(bbox_head_multi_level_kps_0_quant2_threshold[i]);
	}

	// YuNet_Head stride16

	// torch.Size([10, 1, 1, 64])
	for (int i = 0; i < 10 * 1*4; i++) {
		ins.write(bbox_head_multi_level_kps_1_conv1_weight[i]);
	}
	// torch.Size([10, 14])
	for (int i = 0; i < 10 * 14; i++) {
		ins.write(bbox_head_multi_level_kps_1_quant1_threshold[i]);
	}

	// torch.Size([10, 1, 1, 9])
	for (int i = 0; i < 10 * 1; i++) {
		ins.write(bbox_head_multi_level_kps_1_conv2_weight[i]);
	}
	// torch.Size([10, 14])
	for (int i = 0; i < 10 * 14; i++) {
		ins.write(bbox_head_multi_level_kps_1_quant2_threshold[i]);
	}

	// YuNet_Head stride32

	// torch.Size([10, 1, 1, 64])
	for (int i = 0; i < 10 * 1*4; i++) {
		ins.write(bbox_head_multi_level_kps_2_conv1_weight[i]);
	}
	// torch.Size([10, 14])
	for (int i = 0; i < 10 * 14; i++) {
		ins.write(bbox_head_multi_level_kps_2_quant1_threshold[i]);
	}

	// torch.Size([10, 1, 1, 9])
	for (int i = 0; i < 10 * 1; i++) {
		ins.write(bbox_head_multi_level_kps_2_conv2_weight[i]);
	}
	// torch.Size([10, 14])
	for (int i = 0; i < 10 * 14; i++) {
		ins.write(bbox_head_multi_level_kps_2_quant2_threshold[i]);
	}
};
