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
from .hiera_decoder import Hiera_Decoder, Hiera_Conv_Decoder


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

class Hiera_Seg(nn.Module, PyTorchModelHubMixin):
    @has_config
    def __init__(
        self,
        input_size: Tuple[int, ...] = (224, 224),
        in_chans: int = 3,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        num_classes: int = 1000,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, ...] = (2, 2),
        mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
        # mask_unit_attn: which stages use mask unit attention?
        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, ...] = (7, 7),
        patch_stride: Tuple[int, ...] = (4, 4),
        patch_padding: Tuple[int, ...] = (3, 3),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: Union[str, nn.Module] = "LayerNorm",
        head_dropout: float = 0.0,
        head_init_scale: float = 0.001,
        sep_pos_embed: bool = False,
        decoder_type : str = 'vit', 
        use_lora: int = 0,
        deep_supervision: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.patch_kernel = patch_kernel
        self.in_chans = in_chans
        # Do it this way to ensure that the init args are all PoD (for config usage)
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.embed_dim = embed_dim
        depth = sum(stages)
        self.patch_stride = patch_stride
        self.tokens_spatial_shape = [i // s for i, s in zip(input_size, patch_stride)] # 8, 56, 56
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size) # 64
        flat_q_stride = math.prod(q_stride) # 4

        assert q_pool < len(stages)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        self.mask_spatial_shape = [
            i // s for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size) # 8, 7, 7
        ]
        # one mask unit include 8 x 7 x 7 token
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]

        self.patch_embed = PatchEmbed(
            in_chans, embed_dim, patch_kernel, patch_stride, patch_padding
        )
        # 3D: kernel [3,7,7], stride [2,4,4], padding [1,3,3]
        # input 16, 224, 224, output: 8, 56, 56, 96

        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    embed_dim,
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.tokens_spatial_shape[0], embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # Setup roll and reroll modules
        self.unroll = Unroll(
            input_size, patch_stride, [q_stride] * len(self.stage_ends[:-1])
        )
        self.reroll = Reroll(
            input_size,
            patch_stride,
            [q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            q_pool,
        )
        # q_pool locations
        q_pool_blocks = [x + 1 for x in self.stage_ends[:q_pool]]
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        cur_stage = 0
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # Mask unit or global attention.
            # Lag by 1 block, so that global attention,
            # applied post pooling on lower resolution
            use_mask_unit_attn = mask_unit_attn[cur_stage]

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride

            block = HieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.norm = norm_layer(embed_dim)
        # self.head = Head(embed_dim, num_classes, dropout_rate=head_dropout)

        # =========================================================================
        # decoder part 
        # =========================================================================
        if self.embed_dim == 112:
            backbone_channel_list= [896, 448, 224, 112]
        elif self.embed_dim == 144:
            backbone_channel_list= [1152, 576, 288, 144]

        self.decoder_type  = decoder_type
        if self.decoder_type == 'vit':
            blocks_dim_out = [block.dim_out for block in self.blocks]
            self.decoder = Hiera_Decoder(
                input_size=input_size,
                embed_dim=self.embed_dim,
                num_classes=num_classes,
                stages=stages,
                q_pool=q_pool,
                q_stride=q_stride,
                mask_unit_size=mask_unit_size,
                patch_stride=patch_stride,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                use_lora=use_lora,
                deep_supervision=deep_supervision,
                blocks_dim_out=blocks_dim_out,
            )

        elif self.decoder_type == 'fpn':
            self.neck = FpnNeck3D(
                d_model = 256, 
                backbone_channel_list = backbone_channel_list,
                fpn_top_down_levels= None
            )

            self.decoder0_header = \
                nn.Sequential(
                nn.ConvTranspose3d(256, 256 // 2, kernel_size=(2, 2
                , 2), stride=(2, 2, 2)),
                LayerNorm3d(256 // 2),
                torch.nn.GELU(),
                nn.Conv3d(256 // 2, 256 // 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.ConvTranspose3d(256 // 2, 256 // 4, kernel_size=(1, 2, 2), stride=(1, 2, 2),padding=(0, 0, 0)),
                LayerNorm3d(256 // 4),
                torch.nn.GELU(),
                nn.Dropout(0.1, False),
                nn.Conv3d(256 // 4, num_classes, kernel_size=1)
            )
            
        elif self.decoder_type == 'conv':
            # using dpt wise idea

            # target_shape = [[8,14,14],[8,28,28],[8,56,56]]
            print("====================================================")
            print("backbone_channel_list", backbone_channel_list)
            self.decoder=Hiera_Conv_Decoder(in_chans, backbone_channel_list, num_classes)
            # self.decoder4_upsampler = SingleDeconv2DBlock(backbone_channel_list[0], backbone_channel_list[0]//2) # from 8 7 7 to 16 14 14

            # self.decoder3 =  Conv3DBlock(backbone_channel_list[1], backbone_channel_list[1])
            # self.decoder3_upsampler  = \
            # nn.Sequential(
            #     Conv3DBlock(backbone_channel_list[0], backbone_channel_list[1]),
            #     Conv3DBlock(backbone_channel_list[1], backbone_channel_list[1]),
            #     Conv3DBlock(backbone_channel_list[1], backbone_channel_list[1]),
            #     SingleDeconv2DBlock(backbone_channel_list[1], backbone_channel_list[2])
            # )

            # self.decoder2 =  Conv3DBlock(backbone_channel_list[2], backbone_channel_list[2])
            # self.decoder2_upsampler  = \
            #     nn.Sequential(
            #         Conv3DBlock(backbone_channel_list[1], backbone_channel_list[2]),
            #         Conv3DBlock(backbone_channel_list[2], backbone_channel_list[2]),
            #         SingleDeconv2DBlock(backbone_channel_list[2], backbone_channel_list[3])
            #     )  

            # self.decoder1 = Conv3DBlock(backbone_channel_list[3], backbone_channel_list[3])
            # self.decoder1_upsampler  = \
            #     nn.Sequential(
            #         Conv3DBlock(backbone_channel_list[2], backbone_channel_list[3]),
            #         Conv3DBlock(backbone_channel_list[3], backbone_channel_list[3]),
            #         SingleDeconv3DBlock(backbone_channel_list[3], 64)
            #     ) # 16 x 112 x 112
            
            # self.decoder0 = \
            #     nn.Sequential(
            #         Conv3DBlock(in_chans, 32, 3),
            #         Conv3DBlock(32, 64, 3)
            #     )

            # self.decoder0_header = \
            #     nn.Sequential(
            #         Conv3DBlock(128, 64),
            #         Conv3DBlock(64, 64),
            #         SingleConv3DBlock(64, num_classes, 1)
            #     )

        else:
            raise ValueError(f"decoder_type {self.decoder_type} not supported")


        # Initialize everything
        if sep_pos_embed:
            nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
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


    @torch.jit.ignore
    def no_weight_decay(self):
        if self.sep_pos_embed:
            return ["pos_embed_spatial", "pos_embed_temporal"]
        else:
            return ["pos_embed"]

    def get_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        B = x.shape[0]
        # Tokens selected for masking at mask unit level
        num_windows = math.prod(self.mask_spatial_shape)  # num_mask_units
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(B, num_windows, device=x.device)

        # Sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is *keep*, 0 is *remove*
        # Note this is opposite to original MAE
        mask = torch.zeros([B, num_windows], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask.bool()

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


    def forward(self, x):
        input_dim = x.shape[1]
        if input_dim == 1:
            x = x.repeat(1,3,1,1,1)
        x, intermediates = self.forward_encoder(x, 0.0)
        # x shape torch.Size([6, 3, 16, 224, 224])
        pred = self.forward_decoder(x, intermediates)

        return pred

    def forward_decoder(self, x, intermediates):

        if self.decoder_type == 'vit':
            pred = self.decoder(x, intermediates)

        elif self.decoder_type == 'fpn':
            fuse_feat = self.neck(intermediates)
            pred = self.decoder0_header(fuse_feat[0])

        elif self.decoder_type == 'conv':
            pred = self.decoder(x, intermediates)
            # x0 = x
            # x1, x2, x3, x4 = intermediates[0], intermediates[1], intermediates[2], intermediates[3]
            # x1 = x1.permute(0, 4, 1, 2, 3)
            # x2 = x2.permute(0, 4, 1, 2, 3)
            # x3 = x3.permute(0, 4, 1, 2, 3)
            # x4 = x4.permute(0, 4, 1, 2, 3)

            # x4 = self.decoder4_upsampler(x4)
            # x3 = self.decoder3(x3)
            # x3 = self.decoder3_upsampler(torch.cat([x3, x4],dim=1))
            # x2 = self.decoder2(x2)
            # x2 = self.decoder2_upsampler(torch.cat([x2, x3],dim=1))
            # x1 = self.decoder1(x1)
            # x1 = self.decoder1_upsampler(torch.cat([x1, x2],dim=1))
            # x1 = torch.nn.functional.interpolate(x1, size=(16, 224, 224), mode='trilinear')
            # x0 = self.decoder0(x0)
            # pred = self.decoder0_header(torch.cat([x0, x1],dim=1))

        return pred


    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get multi-scale representations from encoder
        _, intermediates = self.forward_hiera(x, None, return_intermediates=True)
        # print("intermediates", len(intermediates), intermediates[0].shape, intermediates[1].shape, intermediates[2].shape, intermediates[3].shape)
        # intermediates 4 torch.Size([6, 8, 56, 56, 112]) torch.Size([6, 8, 28, 28, 224]) torch.Size([6, 8, 14, 14, 448]) torch.Size([6, 8, 7, 7, 896])
        return x, intermediates
    
    def forward_hiera(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_intermediates: bool = True,
    ) -> torch.Tensor:
        """
        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that dim.
        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """
        # Slowfast training passes in a list
        if isinstance(x, list):
            x = x[0]
        intermediates = []

        x = self.patch_embed(
            x,
            mask=mask.view(
                x.shape[0], 1, *self.mask_spatial_shape
            )  # B, C, *mask_spatial_shape
            if mask is not None
            else None,
        )
        x = x + self.get_pos_embed()
        x = self.unroll(x)

        # Discard masked tokens
        if mask is not None:
            x = x[mask[..., None].tile(1, self.mu_size, x.shape[2])].view(
                x.shape[0], -1, x.shape[-1]
            )

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # print("block", i, " x.shape", x.shape) # ([8, 4992, 144])


            if return_intermediates and i in self.stage_ends:
                intermediates.append(self.reroll(x, i, mask=True if self.decoder_type == 'vit' else None))
                # print("intermediates[-1]", intermediates[-1].shape)
                # intermediates[-1] torch.Size([2, 392, 1, 8, 8, 144])
                # intermediates[-1] torch.Size([2, 392, 1, 4, 4, 288])
                # intermediates[-1] torch.Size([2, 392, 1, 2, 2, 576])
                # intermediates[-1] torch.Size([2, 392, 1, 2, 2, 1152])

        # if mask is None:
        #     # x = x.mean(dim=1) # mean pooling
        #     x = self.norm(x)
        #     # x = self.head(x)
        #     if self.decoder_type == 'fpn':
                
        #         x = self.neck(intermediates[1:])

        #         # print(x[0].shape)
        #         # print(x[1].shape)
        #         # print(x[2].shape)
        #         # print(x[3].shape)
        #         x = x[0] # ([6, 256, 8, 56, 56])
        #         x = self.decoder0_header(x)

        #     elif self.decoder_type == 'conv':
        #         x = self.norm(x)
        #         x0, x1, x2, x3, x4 = intermediates[0], intermediates[1], intermediates[2], intermediates[3], intermediates[4]
        #         x1 = x1.permute(0, 4, 1, 2, 3)
        #         x2 = x2.permute(0, 4, 1, 2, 3)
        #         x3 = x3.permute(0, 4, 1, 2, 3)
        #         x4 = x4.permute(0, 4, 1, 2, 3)

        #         # torch.Size([8, 3, 16, 224, 224])
        #         # torch.Size([8, 8, 56, 56, 112])
        #         # torch.Size([8, 8, 28, 28, 224])
        #         # torch.Size([8, 8, 14, 14, 448])
        #         # torch.Size([8, 8, 7, 7, 896])
        #         x4 = self.decoder4_upsampler(x4)
        #         # print("x4", x4.shape) # ([8, 448, 8, 14, 14])
        #         x3 = self.decoder3(x3)
        #         # print("x3", x3.shape) # ([8, 448, 8, 14, 14])
        #         x3 = self.decoder3_upsampler(torch.cat([x3, x4],dim=1))
        #         # print("x3 upsample", x3.shape) # ([8, 224, 8, 28, 28])
        #         x2 = self.decoder2(x2)
        #         # print("x2", x2.shape) # ([8, 224, 8, 28, 28])
        #         x2 = self.decoder2_upsampler(torch.cat([x2, x3],dim=1))
        #         # print("x2 upsample", x2.shape) # ([8, 224, 8, 28, 28])
        #         x1 = self.decoder1(x1)
        #         # print("x1", x1.shape) # [8, 112, 8, 56, 56])
        #         x1 = self.decoder1_upsampler(torch.cat([x1, x2],dim=1))
        #         # print("x1 upsample", x1.shape) # ([8, 64, 8, 112, 112])
        #         x1 = torch.nn.functional.interpolate(x1, size=(16, 224, 224), mode='trilinear')
        #        #  print("x1 further upsample", x1.shape) # e([8, 64, 16, 224, 224])
        #         x0 = self.decoder0(x0)
        #         # print("x0", x0.shape)
        #         x = self.decoder0_header(torch.cat([x0, x1],dim=1))
        #         # print("x", x.shape)
                

        # x may not always be in spatial order here.
        # e.g. if q_pool = 2, mask_unit_size = (8, 8), and
        # q_stride = (2, 2), not all unrolls were consumed,
        # intermediates[-1] is x in spatial order
        if return_intermediates:
            return x, intermediates

        return x


# Image models

@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_tiny_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_small_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_small_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_base_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_base_plus_224(**kwdargs):
    return Hiera(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_large_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_large_224(**kwdargs):
    return Hiera(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs)


@pretrained_model({
    "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_224.pth",
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_224.pth",
}, default="mae_in1k_ft_in1k")
def hiera_huge_224(**kwdargs):
    return Hiera(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs)


# Video models

@pretrained_model({
    "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth",
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth",
}, default="mae_k400_ft_k400")
def hiera_base_16x224(num_classes: int = 400, **kwdargs):
    return Hiera(
        num_classes=num_classes,  # K400 has 400 classes
        input_size=(16, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_pos_embed=True,
        **kwdargs
    )


@pretrained_model({
    "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_16x224.pth",
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_16x224.pth",
}, default="mae_k400_ft_k400")
def hiera_base_plus_16x224(**kwdargs):
    return hiera_base_16x224(
        embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs
    )


@pretrained_model({
    "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_large_16x224.pth",
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_16x224.pth",
}, default="mae_k400_ft_k400")
def hiera_large_16x224(**kwdargs):
    return hiera_base_16x224(
        embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs
    )


@pretrained_model({
    "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_16x224.pth",
    "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_16x224.pth",
}, default="mae_k400_ft_k400")
def hiera_huge_16x224(**kwdargs):
    return hiera_base_16x224(
        embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs
    )