# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
#
# Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan,
# Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed,
# Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer.
#
# Paper: https://arxiv.org/abs/2306.00989/
#
# References:
# slowfast: https://github.com/facebookresearch/SlowFast
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import math
from functools import partial
from typing import List, Tuple, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, Mlp

from .hiera_utils import pretrained_model, conv_nd, do_pool, do_masked_conv, Unroll, Reroll, undo_windowing 
from .hfhub import has_config, PyTorchModelHubMixin
from .neck import FpnNeck3D, FpnNeck
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.videomae.video_vit import LayerNorm3d
from .hiera_mae import apply_fusion_head
from .hiera import  HieraBlock, Head, PatchEmbed, Hiera



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


class Hiera_Conv_Decoder(nn.Module):
    def __init__(
        self,
        in_chans,
        backbone_channel_list: List[int],
        num_classes: int = 1000,
    ):

        super().__init__()
        self.decoder4_upsampler = SingleDeconv2DBlock(backbone_channel_list[0], backbone_channel_list[0]//2) # from 8 7 7 to 16 14 14

        self.decoder3 =  Conv3DBlock(backbone_channel_list[1], backbone_channel_list[1])
        self.decoder3_upsampler  = \
        nn.Sequential(
            Conv3DBlock(backbone_channel_list[0], backbone_channel_list[1]),
            Conv3DBlock(backbone_channel_list[1], backbone_channel_list[1]),
            Conv3DBlock(backbone_channel_list[1], backbone_channel_list[1]),
            SingleDeconv2DBlock(backbone_channel_list[1], backbone_channel_list[2])
        )

        self.decoder2 =  Conv3DBlock(backbone_channel_list[2], backbone_channel_list[2])
        self.decoder2_upsampler  = \
            nn.Sequential(
                Conv3DBlock(backbone_channel_list[1], backbone_channel_list[2]),
                Conv3DBlock(backbone_channel_list[2], backbone_channel_list[2]),
                SingleDeconv2DBlock(backbone_channel_list[2], backbone_channel_list[3])
            )  

        self.decoder1 = Conv3DBlock(backbone_channel_list[3], backbone_channel_list[3])
        self.decoder1_upsampler  = \
            nn.Sequential(
                Conv3DBlock(backbone_channel_list[2], backbone_channel_list[3]),
                Conv3DBlock(backbone_channel_list[3], backbone_channel_list[3]),
                SingleDeconv3DBlock(backbone_channel_list[3], 64)
            ) # 16 x 112 x 112
        
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


    def reshape_intermediates(self, x):
        B, num_mask_units = x.shape[0:2]


        return x
        

    def forward(self, x, intermediates):
        # print('='*20)
        # print("x", x.shape)
        # x torch.Size([2, 3, 16, 224, 224])
        # ====================
        # x1 torch.Size([2, 392, 1, 8, 8, 144])
        x0 = x
        x1, x2, x3, x4 = intermediates[0], intermediates[1], intermediates[2], intermediates[3]
        # x1 = self.reshape_intermediates(x1)
        # x2 = self.reshape_intermediates(x2)
        # x3 = self.reshape_intermediates(x3)
        # x4 = self.reshape_intermediates(x4)
        x1 = x1.permute(0, 4, 1, 2, 3)
        x2 = x2.permute(0, 4, 1, 2, 3)
        x3 = x3.permute(0, 4, 1, 2, 3)
        x4 = x4.permute(0, 4, 1, 2, 3)


        # print('='*30)
        # print("x1", x1.shape)
        # intermediates[-1] torch.Size([2, 8, 56, 56, 144])
        # intermediates[-1] torch.Size([2, 8, 28, 28, 288])
        # intermediates[-1] torch.Size([2, 8, 14, 14, 576])
        # intermediates[-1] torch.Size([2, 8, 14, 14, 1152])



        x4 = self.decoder4_upsampler(x4)
        x3 = self.decoder3(x3)
        x3 = self.decoder3_upsampler(torch.cat([x3, x4],dim=1))
        x2 = self.decoder2(x2)
        x2 = self.decoder2_upsampler(torch.cat([x2, x3],dim=1))
        x1 = self.decoder1(x1)
        x1 = self.decoder1_upsampler(torch.cat([x1, x2],dim=1))
        x1 = torch.nn.functional.interpolate(x1, size=(16, 224, 224), mode='trilinear')
        x0 = self.decoder0(x0)
        pred = self.decoder0_header(torch.cat([x0, x1],dim=1))

        return pred



class Hiera_Decoder(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, ...] = (224, 224),
        embed_dim: int = 96,  # initial embed dim
        num_classes: int = 1000,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, ...] = (2, 2),
        mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
        # mask_unit_attn: which stages use mask unit attention?
        patch_stride: Tuple[int, ...] = (4, 4),
        mlp_ratio: float = 4.0,
        norm_layer: Union[str, nn.Module] = "LayerNorm",
        use_lora: int = 0,
        deep_supervision: bool = False,
        blocks_dim_out: List[int] = [96, 192, 384, 768],
        **kwargs,
    ):
        super().__init__()

        # Do it this way to ensure that the init args are all PoD (for config usage)
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.tokens_spatial_shape = [i // s for i, s in zip(input_size, patch_stride)] # 8, 56, 56
        flat_mu_size = math.prod(mask_unit_size) # 64
 

        assert q_pool < len(stages)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
 
        # one mask unit include 8 x 7 x 7 token
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]



        # self.head = Head(embed_dim, num_classes, dropout_rate=head_dropout)

        # =========================================================================
        # decoder part 
        # =========================================================================
        
        if self.embed_dim == 112:
            backbone_channel_list= [896, 448, 224, 112]
        elif self.embed_dim == 144:
            backbone_channel_list= [1152, 576, 288, 144]

        
        decoder_embed_dim = 512
        decoder_num_heads = 16
        decoder_depth = 8

        print("blocks_dim_out", blocks_dim_out)

        encoder_dim_out = blocks_dim_out[-1]
        self.encoder_norm = norm_layer(encoder_dim_out)
        self.decoder_embed = nn.Linear(encoder_dim_out, decoder_embed_dim)
        self.mask_unit_spatial_shape_final = [
            i // s ** (self.q_pool) for i, s in zip(self.mask_unit_size, self.q_stride)
        ]
        self.tokens_spatial_shape_final = [
            i // s ** (self.q_pool)
            for i, s in zip(self.tokens_spatial_shape, self.q_stride)
        ]
        # --------------------------------------------------------------------------
        # Multi-scale fusion heads
        curr_mu_size = self.mask_unit_size
        self.multi_scale_fusion_heads = nn.ModuleList()

        for i in self.stage_ends[: self.q_pool]:  # resolution constant after q_pool
            kernel = [
                i // s for i, s in zip(curr_mu_size, self.mask_unit_spatial_shape_final)
            ]
            curr_mu_size = [i // s for i, s in zip(curr_mu_size, self.q_stride)]
            self.multi_scale_fusion_heads.append(
                conv_nd(len(self.q_stride))(
                    blocks_dim_out[i],
                    encoder_dim_out,
                    kernel_size=kernel,
                    stride=kernel,
                )
            )
        self.multi_scale_fusion_heads.append(nn.Identity())  # final stage, no transform

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1, math.prod(self.tokens_spatial_shape_final), decoder_embed_dim
            )
        )

        self.decoder_blocks = nn.ModuleList(
            [
                HieraBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_stride = patch_stride[-1] * (
            self.q_stride[-1] ** self.q_pool
        )  # patch stride of prediction

        self.seg_pred = nn.Linear(
            decoder_embed_dim,
            (self.pred_stride ** min(2, len(self.q_stride))) * 2  * num_classes,
        )  # predictor
        # --------------------------------------------------------------------------

        self.apply(partial(self._init_weights))
        # self.head.projection.weight.data.mul_(head_init_scale)
        # self.head.projection.bias.data.mul_(head_init_scale)


    def _init_weights(self, m, init_bias=0.02):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, init_bias)
            nn.init.constant_(m.weight, 1.0)
            nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.sep_pos_embed:
            return ["pos_embed_spatial", "pos_embed_temporal"]
        else:
            return ["pos_embed"]

    def get_pos_embed(self) -> torch.Tensor:
        if self.sep_pos_embed:
            return self.pos_embed_spatial.repeat(
                1, self.tokens_spatial_shape[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                dim=1,
            )
        else:
            return self.pos_embed


    def forward(self, x, intermediates):
        pred = self.forward_decoder(x, intermediates)

        return pred

    def forward_decoder(self, x, intermediates):

        intermediates = intermediates[: self.q_pool] + intermediates[-1:]

        x = 0.0
        for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates):
            # print("interm_x", interm_x.shape)
            # interm_x torch.Size([6, 8, 56, 56, 112])
            feat = apply_fusion_head(head, interm_x)
            # print("feat", feat.shape)
            x += feat

        x = self.encoder_norm(x)
        x = self.decoder_embed(x)

        x = undo_windowing(
            x,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
            )
        # print("x", x.shape) # x torch.Size([6, 8, 14, 14, 512])
        B,D,H,W,C = x.shape
        # Add pos embed
        # flatten tokens 
        x = x.view(x.shape[0], -1, x.shape[-1])
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        pred = self.seg_pred(x) # ([6, 1568, 768])
        pred = pred.view(pred.shape[0], D, H, W, 2, 16,16, -1)
        pred = torch.einsum("nthwupqc->nctuhpwq", pred)
        pred = pred.reshape(B, -1, 2*D, H*16, W*16)

        return pred


