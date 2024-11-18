# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from .videomae import video_vit
from .videomae.video_vit import LayerNorm3d
# from .videomae.logging import master_print as print

import torch.nn.functional as F
import numpy as np 
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0


class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=(1,2,2), stride=(1,2,2), padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)

class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)

class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()


        self.conv1 = nn.Conv3d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv3d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # if self.bn == True:
        #     self.bn1 = nn.BatchNorm3d(features)
        #     self.bn2 = nn.BatchNorm3d(features)
        self.layern1 = LayerNorm3d(features)
        self.layern2 = LayerNorm3d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        out=self.layern1(out)

        out = self.activation(out)
        out = self.conv2(out)
        out=self.layern2(out)


        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        feature_align_shape,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.feature_align_shape = feature_align_shape

        out_features = features

        self.out_conv = nn.Conv3d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        # if self.scales > 1:
        #     print("output size", output.size())
        #     output = nn.functional.interpolate(
        #         output, scale_factor=(2,2,2), mode="area",
        #     )
        # else:
        #     output = nn.functional.interpolate(
        #         output, scale_factor=(1,2,2), mode="area", 
        #     )
        output = nn.functional.interpolate(output, size=self.feature_align_shape, mode="area")
        # print("interpolated output size", output.size())

        output = self.out_conv(output)
        # print("conv output size", output.size())

        return output
    
def _make_fusion_block(features, feature_align_shape):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        feature_align_shape=feature_align_shape,
    )


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]

class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper


class ConvDecoder(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        in_chans=3,
        embed_dim=1024,
        num_classes=1,
        num_frames=16,
        input_size = [8, 14, 14],
        deep_supervision=False,
        **kwargs,
    ):
        super().__init__()
        # cls_embed = True
        self.deep_supervision = deep_supervision
        self.input_size = input_size
        self.output_size = [num_frames,img_size, img_size]
        self.embed_dim = embed_dim
        self.multi_feat_layer = [5,11,17]
        # print("self.input_size", self.input_size) #(8, 14, 14)

        self.decoder24_upsampler = SingleDeconv3DBlock(embed_dim, 512)

        self.decoder17 = \
        Deconv3DBlock(embed_dim, 512)

        self.decoder17_upsampler = \
        nn.Sequential(
            Conv3DBlock(1024, 512),
            Conv3DBlock(512, 512),
            Conv3DBlock(512, 512),
            SingleDeconv2DBlock(512, 256)
        )

        self.decoder11 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv2DBlock(512, 256),
            )

        self.decoder11_upsampler = \
            nn.Sequential(
                Conv3DBlock(512, 256),
                Conv3DBlock(256, 256),
                SingleDeconv2DBlock(256, 128)
            )  

        self.decoder5 = \
        nn.Sequential(
            Deconv3DBlock(embed_dim, 512),
            Deconv2DBlock(512, 256),
            Deconv2DBlock(256, 128)
        )          
        
        self.decoder5_upsampler = \
            nn.Sequential(
                Conv3DBlock(256, 128),
                Conv3DBlock(128, 128),
                SingleDeconv2DBlock(128, 64)
            )

        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(in_chans, 32, 3),
                Conv3DBlock(32, 64, 3)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SingleConv3DBlock(64, num_classes, 1)
            )

        self.decoder0_header_aux = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SingleConv3DBlock(64, num_classes, 1)
            )

        

    # @staticmethod
    # def initialize(module):
    #     InitWeights_He(1e-3)(module)
        # init_last_bn_before_add_to_0(module)


    def forward(self, multi_scale_feat):
        # x (4, 1568, 1024)
        # print("sample_t", t.size())

        # N = x.shape[0]
        # T = self.patch_embed.t_grid_size
        # H = W = self.patch_embed.grid_size

        # embed tokens
        # C = x.shape[-1]
        seg_outputs = []
        x0, x5, x11, x17, x24 = multi_scale_feat
        x5 = self.convert_to_3d_tensor(x5)
        x11 = self.convert_to_3d_tensor(x11)
        x17 = self.convert_to_3d_tensor(x17)
        x24 = self.convert_to_3d_tensor(x24)

        x24 = self.decoder24_upsampler(x24)
        x17 = self.decoder17(x17)

        # print("x17", x17.size())
        # torch.Size([2, 512, 16, 28, 28])

        x17 = self.decoder17_upsampler(torch.cat([x17, x24],dim=1))
        x11 = self.decoder11(x11)

        # print("x11", x11.size())
        # torch.Size([2, 256, 16, 56, 56])
        x11 = self.decoder11_upsampler(torch.cat([x11, x17], dim=1))
        x5 = self.decoder5(x5)
        if self.deep_supervision:
            aux_output = self.decoder0_header_aux(x5)
            seg_outputs.append(aux_output)

        # print("x5", x5.size())
        # torch.Size([2, 128, 16, 112, 112])
        x5 = self.decoder5_upsampler(torch.cat([x5, x11], dim=1))
        x0 = self.decoder0(x0)

        # print("x0", x0.size())
        # torch.Size([2, 64, 16, 224, 224])
        x = self.decoder0_header(torch.cat([x0, x5], dim=1))
        seg_outputs.append(x)
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs

        return r

    def convert_to_3d_tensor(self, x):
        # TODO use_readout
        # if self.use_readout == 'ignore' and self.cls_embed :
        #     x = x[:, 1:]
        N = x.shape[0]
        C = x.shape[-1]
        # print("x size", x.size())
        x = x.view([N, self.input_size[0],self.input_size[1], self.input_size[2], C]) # B, 8, 14, 14, 512
        x = x.permute(0, 4, 1, 2, 3) # B, 1024, 8, 14, 14 
        return x


    # def forward(self, imgs, t=None):
    #     _ = self.patchify(imgs)
    #     latent, multi_scale_feat = self.forward_encoder(imgs)
    #     pred = self.forward_decoder(latent, t=t, multi_scale_feat=multi_scale_feat)
    #     return pred



