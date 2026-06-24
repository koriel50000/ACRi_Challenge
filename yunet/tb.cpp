#include "kernel.hpp"
#include "image.hpp"
#include "params.hpp"
#include "output.hpp"
#include <cstdlib>

void input_stream(hls::stream<axis_data64>& ins);
int validate_output_stream(hls::stream<axis_data8>& outs);

int main(int argc, char** argv)
{
	hls::stream<axis_data64> ins;
	hls::stream<axis_data8> outs;

	input_stream(ins);
	kernel(ins, outs);
	return validate_output_stream(outs);
}

void input_stream(hls::stream<axis_data64>& ins) {
	axis_data64 pkt;
	pkt.last = 0;

	// torch.Size([1, 3, 160, 160])
	for (int i = 0; i < 160 * 160; i++) {
		pkt.data = images[i];
		ins.write(pkt);
	}

	// YuNetBackbone stage0
	// Conv_head

	// torch.Size([16, 3, 3, 3])
	for (int i = 0; i < 16 * 9; i++) {
		pkt.data = backbone_model0_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([16, 7])
	for (int i = 0; i < 16 * 4; i++) {
		pkt.data = backbone_model0_relu1_threshold[i];
		ins.write(pkt);
	}

	// Conv_head ConvDPUnit

	// torch.Size([16, 1, 1, 16])
	for (int i = 0; i < 16 * 1; i++) {
		pkt.data = backbone_model0_conv2_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([16, 14])
	for (int i = 0; i < 16 * 4; i++) {
		pkt.data = backbone_model0_conv2_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([16, 1, 1, 9])
	for (int i = 0; i < 16 * 1; i++) {
		pkt.data = backbone_model0_conv2_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([16, 7])
	for (int i = 0; i < 16 * 4; i++) {
		pkt.data = backbone_model0_conv2_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNetBackbone stage1
	// YuNetBackbone Conv4layerBlock 1

	// torch.Size([16, 1, 1, 16])
	for (int i = 0; i < 16 * 1; i++) {
		pkt.data = backbone_model1_conv1_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([16, 14])
	for (int i = 0; i < 16 * 4; i++) {
		pkt.data = backbone_model1_conv1_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([16, 1, 1, 9])
	for (int i = 0; i < 16 * 1; i++) {
		pkt.data = backbone_model1_conv1_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([16, 7])
	for (int i = 0; i < 16 * 4; i++) {
		pkt.data = backbone_model1_conv1_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNetBackbone Conv4layerBlock 2

	// torch.Size([64, 1, 1, 16])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = backbone_model1_conv2_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model1_conv2_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = backbone_model1_conv2_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model1_conv2_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNetBackbone stage2
	// YuNetBackbone Conv4layerBlock 1

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = backbone_model2_conv1_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model2_conv1_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = backbone_model2_conv1_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model2_conv1_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNetBackbone Conv4layerBlock 2

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = backbone_model2_conv2_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model2_conv2_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = backbone_model2_conv2_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model2_conv2_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNetBackbone stage3
	// YuNetBackbone Conv4layerBlock 1

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = backbone_model3_conv1_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model3_conv1_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = backbone_model3_conv1_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model3_conv1_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNetBackbone Conv4layerBlock 2

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = backbone_model3_conv2_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model3_conv2_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = backbone_model3_conv2_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model3_conv2_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNetBackbone stage4
	// YuNetBackbone Conv4layerBlock 1

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = backbone_model4_conv1_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model4_conv1_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = backbone_model4_conv1_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model4_conv1_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNetBackbone Conv4layerBlock 2

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = backbone_model4_conv2_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model4_conv2_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = backbone_model4_conv2_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model4_conv2_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNetBackbone stage5
	// YuNetBackbone Conv4layerBlock 1

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = backbone_model5_conv1_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model5_conv1_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = backbone_model5_conv1_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model5_conv1_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNetBackbone Conv4layerBlock 2

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = backbone_model5_conv2_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model5_conv2_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = backbone_model5_conv2_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = backbone_model5_conv2_relu2_threshold[i];
		ins.write(pkt);
	}

	// TFPN stride32
	// TFPN ConvDPUnit

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = neck_lateral_convs_2_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = neck_lateral_convs_2_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = neck_lateral_convs_2_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = neck_lateral_convs_2_relu2_threshold[i];
		ins.write(pkt);
	}

	// TFPN stride16
	// TFPN ConvDPUnit

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = neck_lateral_convs_1_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = neck_lateral_convs_1_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = neck_lateral_convs_1_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = neck_lateral_convs_1_relu2_threshold[i];
		ins.write(pkt);
	}

	// TFPN stride8
	// TFPN ConvDPUnit

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = neck_lateral_convs_0_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = neck_lateral_convs_0_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = neck_lateral_convs_0_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = neck_lateral_convs_0_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head stride8
	// YuNet_Head shared ConvDPUnit

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_share_convs_0_0_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = bbox_head_multi_level_share_convs_0_0_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = bbox_head_multi_level_share_convs_0_0_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = bbox_head_multi_level_share_convs_0_0_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head stride16
	// YuNet_Head shared ConvDPUnit

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_share_convs_1_0_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = bbox_head_multi_level_share_convs_1_0_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = bbox_head_multi_level_share_convs_1_0_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = bbox_head_multi_level_share_convs_1_0_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head stride32
	// YuNet_Head shared ConvDPUnit

	// torch.Size([64, 1, 1, 64])
	for (int i = 0; i < 64 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_share_convs_2_0_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 14])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = bbox_head_multi_level_share_convs_2_0_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([64, 1, 1, 9])
	for (int i = 0; i < 64 * 1; i++) {
		pkt.data = bbox_head_multi_level_share_convs_2_0_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([64, 7])
	for (int i = 0; i < 64 * 4; i++) {
		pkt.data = bbox_head_multi_level_share_convs_2_0_relu2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head cls ConvDPUnit
	// YuNet_Head stride8

	// torch.Size([1, 1, 1, 64])
	for (int i = 0; i < 1 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_cls_0_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 4; i++) {
		pkt.data = bbox_head_multi_level_cls_0_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([1, 1, 1, 9])
	for (int i = 0; i < 1 * 1; i++) {
		pkt.data = bbox_head_multi_level_cls_0_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 4; i++) {
		pkt.data = bbox_head_multi_level_cls_0_quant2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head stride16

	// torch.Size([1, 1, 1, 64])
	for (int i = 0; i < 1 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_cls_1_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 4; i++) {
		pkt.data = bbox_head_multi_level_cls_1_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([1, 1, 1, 9])
	for (int i = 0; i < 1 * 1; i++) {
		pkt.data = bbox_head_multi_level_cls_1_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 4; i++) {
		pkt.data = bbox_head_multi_level_cls_1_quant2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head stride32

	// torch.Size([1, 1, 1, 64])
	for (int i = 0; i < 1 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_cls_2_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 4; i++) {
		pkt.data = bbox_head_multi_level_cls_2_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([1, 1, 1, 9])
	for (int i = 0; i < 1 * 1; i++) {
		pkt.data = bbox_head_multi_level_cls_2_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 4; i++) {
		pkt.data = bbox_head_multi_level_cls_2_quant2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head bbox ConvDPUnit
	// YuNet_Head stride8

	// torch.Size([4, 1, 1, 64])
	for (int i = 0; i < 4 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_bbox_0_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([4, 14])
	for (int i = 0; i < 4 * 4; i++) {
		pkt.data = bbox_head_multi_level_bbox_0_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([4, 1, 1, 9])
	for (int i = 0; i < 4 * 1; i++) {
		pkt.data = bbox_head_multi_level_bbox_0_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([4, 14])
	for (int i = 0; i < 4 * 4; i++) {
		pkt.data = bbox_head_multi_level_bbox_0_quant2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head stride16

	// torch.Size([4, 1, 1, 64])
	for (int i = 0; i < 4 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_bbox_1_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([4, 14])
	for (int i = 0; i < 4 * 4; i++) {
		pkt.data = bbox_head_multi_level_bbox_1_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([4, 1, 1, 9])
	for (int i = 0; i < 4 * 1; i++) {
		pkt.data = bbox_head_multi_level_bbox_1_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([4, 14])
	for (int i = 0; i < 4 * 4; i++) {
		pkt.data = bbox_head_multi_level_bbox_1_quant2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head stride32

	// torch.Size([4, 1, 1, 64])
	for (int i = 0; i < 4 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_bbox_2_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([4, 14])
	for (int i = 0; i < 4 * 4; i++) {
		pkt.data = bbox_head_multi_level_bbox_2_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([4, 1, 1, 9])
	for (int i = 0; i < 4 * 1; i++) {
		pkt.data = bbox_head_multi_level_bbox_2_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([4, 14])
	for (int i = 0; i < 4 * 4; i++) {
		pkt.data = bbox_head_multi_level_bbox_2_quant2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head obj ConvDPUnit
	// YuNet_Head stride8

	// torch.Size([1, 1, 1, 64])
	for (int i = 0; i < 1 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_obj_0_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 4; i++) {
		pkt.data = bbox_head_multi_level_obj_0_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([1, 1, 1, 9])
	for (int i = 0; i < 1 * 1; i++) {
		pkt.data = bbox_head_multi_level_obj_0_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 4; i++) {
		pkt.data = bbox_head_multi_level_obj_0_quant2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head stride16

	// torch.Size([1, 1, 1, 64])
	for (int i = 0; i < 1 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_obj_1_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 4; i++) {
		pkt.data = bbox_head_multi_level_obj_1_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([1, 1, 1, 9])
	for (int i = 0; i < 1 * 1; i++) {
		pkt.data = bbox_head_multi_level_obj_1_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 4; i++) {
		pkt.data = bbox_head_multi_level_obj_1_quant2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head stride32

	// torch.Size([1, 1, 1, 64])
	for (int i = 0; i < 1 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_obj_2_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 4; i++) {
		pkt.data = bbox_head_multi_level_obj_2_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([1, 1, 1, 9])
	for (int i = 0; i < 1 * 1; i++) {
		pkt.data = bbox_head_multi_level_obj_2_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([1, 14])
	for (int i = 0; i < 1 * 4; i++) {
		pkt.data = bbox_head_multi_level_obj_2_quant2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head kps ConvDPUnit
	// YuNet_Head stride8

	// torch.Size([10, 1, 1, 64])
	for (int i = 0; i < 10 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_kps_0_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([10, 14])
	for (int i = 0; i < 10 * 4; i++) {
		pkt.data = bbox_head_multi_level_kps_0_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([10, 1, 1, 9])
	for (int i = 0; i < 10 * 1; i++) {
		pkt.data = bbox_head_multi_level_kps_0_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([10, 14])
	for (int i = 0; i < 10 * 4; i++) {
		pkt.data = bbox_head_multi_level_kps_0_quant2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head stride16

	// torch.Size([10, 1, 1, 64])
	for (int i = 0; i < 10 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_kps_1_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([10, 14])
	for (int i = 0; i < 10 * 4; i++) {
		pkt.data = bbox_head_multi_level_kps_1_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([10, 1, 1, 9])
	for (int i = 0; i < 10 * 1; i++) {
		pkt.data = bbox_head_multi_level_kps_1_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([10, 14])
	for (int i = 0; i < 10 * 4; i++) {
		pkt.data = bbox_head_multi_level_kps_1_quant2_threshold[i];
		ins.write(pkt);
	}

	// YuNet_Head stride32

	// torch.Size([10, 1, 1, 64])
	for (int i = 0; i < 10 * 1*4; i++) {
		pkt.data = bbox_head_multi_level_kps_2_conv1_weight[i];
		ins.write(pkt);
	}
	// torch.Size([10, 14])
	for (int i = 0; i < 10 * 4; i++) {
		pkt.data = bbox_head_multi_level_kps_2_quant1_threshold[i];
		ins.write(pkt);
	}

	// torch.Size([10, 1, 1, 9])
	for (int i = 0; i < 10 * 1; i++) {
		pkt.data = bbox_head_multi_level_kps_2_conv2_weight[i];
		ins.write(pkt);
	}
	// torch.Size([10, 14])
	for (int i = 0; i < 10 * 4; i++) {
		pkt.data = bbox_head_multi_level_kps_2_quant2_threshold[i];
		ins.write(pkt);
	}
}

int validate_output_stream(hls::stream<axis_data8>& outs) {
	int size = outs.read().data;
	// printf("size=%d\n", size);
	for (int i = 0; i < size; i++) {
		uint8_t x1 = outs.read().data;
		uint8_t y1 = outs.read().data;
		uint8_t x2 = outs.read().data;
		uint8_t y2 = outs.read().data;
		uint8_t hi = outs.read().data;
		uint8_t lo = outs.read().data;
		uint16_t score = (hi << 8) | lo;
		uint8_t  kps[10];
		for (int j = 0; j < 10; j++) {
			kps[j] = outs.read().data;
		}

      if (x1 != expected_box_score[i][0])
            return EXIT_FAILURE;
        if (y1 != expected_box_score[i][1])
            return EXIT_FAILURE;
        if (x2 != expected_box_score[i][2])
            return EXIT_FAILURE;
        if (y2 != expected_box_score[i][3])
            return EXIT_FAILURE;
        if (score != expected_box_score[i][4])
            return EXIT_FAILURE;

        for (int j = 0; j < 10; j++)
            if (kps[j] != expected_kps[i][j])
                return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

// int validate_output_stream(hls::stream<axis_data>& outs) {
// 	// YuNet_Head cls ConvDPUnit
// 	// YuNet_Head stride8
// 	for (int i = 0; i < 20 * 20; i++) {
// 		if (outs.read().data != cls_scores_stride8[i])
// 			return EXIT_FAILURE;
// 	}
// 	// YuNet_Head stride16
// 	for (int i = 0; i < 10 * 10; i++) {
// 		if (outs.read().data != cls_scores_stride16[i])
// 			return EXIT_FAILURE;
// 	}
// 	// YuNet_Head stride32
// 	for (int i = 0; i < 5 * 5; i++) {
// 		if (outs.read().data != cls_scores_stride32[i])
// 			return EXIT_FAILURE;
// 	}

// 	// YuNet_Head bbox ConvDPUnit
// 	// YuNet_Head stride8
// 	for (int i = 0; i < 20 * 20; i++) {
// 		if (outs.read().data != bbox_preds_stride8[i])
// 			return EXIT_FAILURE;
// 	}
// 	// YuNet_Head stride16
// 	for (int i = 0; i < 10 * 10; i++) {
// 		if (outs.read().data != bbox_preds_stride16[i])
// 			return EXIT_FAILURE;
// 	}
// 	// YuNet_Head stride32
// 	for (int i = 0; i < 5 * 5; i++) {
// 		if (outs.read().data != bbox_preds_stride32[i])
// 			return EXIT_FAILURE;
// 	}

// 	// YuNet_Head obj ConvDPUnit
// 	// YuNet_Head stride8
// 	for (int i = 0; i < 20 * 20; i++) {
// 		if (outs.read().data != objectnesses_stride8[i])
// 			return EXIT_FAILURE;
// 	}
// 	// YuNet_Head stride16
// 	for (int i = 0; i < 10 * 10; i++) {
// 		if (outs.read().data != objectnesses_stride16[i])
// 			return EXIT_FAILURE;
// 	}
// 	// YuNet_Head stride32
// 	for (int i = 0; i < 5 * 5; i++) {
// 		if (outs.read().data != objectnesses_stride32[i])
// 			return EXIT_FAILURE;
// 	}

// 	// YuNet_Head kps ConvDPUnit
// 	// YuNet_Head stride8
// 	for (int i = 0; i < 20 * 20; i++) {
// 		if (outs.read().data != kps_preds_stride8[i])
// 			return EXIT_FAILURE;
// 	}
// 	// YuNet_Head stride16
// 	for (int i = 0; i < 10 * 10; i++) {
// 		if (outs.read().data != kps_preds_stride16[i])
// 			return EXIT_FAILURE;
// 	}
// 	// YuNet_Head stride32
// 	for (int i = 0; i < 5 * 5; i++) {
// 		if (outs.read().data != kps_preds_stride32[i])
// 			return EXIT_FAILURE;
// 	}

// 	return EXIT_SUCCESS;
// }
