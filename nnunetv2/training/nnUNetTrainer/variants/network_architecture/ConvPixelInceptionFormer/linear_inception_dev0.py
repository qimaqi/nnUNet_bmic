from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
from typing import Union, Type, List, Tuple, Dict, Any, Callable
from collections import OrderedDict
from functools import partial

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep

from timm.models.vision_transformer import DropPath # ,  Mlp
from torch import nn, einsum
import time
import numpy as np
from torchvision.models.vision_transformer import EncoderBlock
from torch.nn import LayerNorm
import json 
import omegaconf
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

from .global_attention import Mlp
from .utils import PositionalEncoding1D ,PositionalEncoding3D
import loralib as lora
from .basic_unet import TwoConv

torch.autograd.detect_anomaly(check_nan=True)




class ConvPixelInceptionDecoder_2Layer(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 32),
        patch_size: Sequence[int] = (20, 256, 224),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("GROUP", {"num_groups": 1, "affine": False}),  #("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        attn_dict: dict = None,
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 4)
        print(f"BasicUNet features: {fea}.")

        self.upcat_2 = UpCat_ConvPixelFormer(
            spatial_dims=spatial_dims, in_chns=fea[2], 
            cat_chns=fea[1], out_chns=fea[1], 
            act=act, 
            norm=norm, 
            bias=bias, 
            dropout=dropout, 
            upsample=upsample)

        self.upcat_1 = UpCat_ConvPixelFormer(
            spatial_dims=spatial_dims, 
            in_chns=fea[1], cat_chns=fea[0], out_chns=fea[-1], act=act, norm=norm, bias=bias, dropout=dropout, upsample=upsample, halves=False)
        
        # first 2 layer
        feat_shape_0 = [patch_size[0], patch_size[1], patch_size[2]]
        feat_shape_1 = [patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]
        feat_shape_2 = [patch_size[0] // 4, patch_size[1] // 4, patch_size[2] // 4]

        self.layer2 = LinearInceptionAtnnBlock(
            dim=fea[1], num_heads=attn_dict['head_dims'][0],
            Inception_window_size=[[4, 16, 16]],
            qk_dim_compresion=attn_dict['qk_dim_compresion'][0],
            qk_nonlin=attn_dict['qk_nonlin'],
            feat_shape=feat_shape_1,
            debug=attn_dict.get('debug', False)
        )


        self.final_conv = Conv["conv", spatial_dims](fea[-1], out_channels, kernel_size=1)

    def forward(self, x0, x1, x2):

        u2 = self.upcat_2(x2, x1)

        with torch.cuda.amp.autocast(enabled=False):
            u2 = u2.to(dtype=torch.float32)
            u2 = self.layer2(u2)

        u1 = self.upcat_1(u2, x0)
        logits = self.final_conv(u1)
        return logits
    
class ConvPixelInceptionConvDecoder_2Layer(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 32),
        patch_size: Sequence[int] = (20, 256, 224),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("GROUP", {"num_groups": 1, "affine": False}),  #("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        attn_dict: dict = None,
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 4)
        print(f"BasicUNet features: {fea}.")

        self.upcat_2 = UpCat_ConvPixelFormer(
            spatial_dims=spatial_dims, in_chns=fea[2], 
            cat_chns=fea[1], out_chns=fea[1], 
            act=act, 
            norm=norm, 
            bias=bias, 
            dropout=dropout, 
            upsample=upsample)

        self.upcat_1 = UpCat_ConvPixelFormer(
            spatial_dims=spatial_dims, 
            in_chns=fea[1], cat_chns=fea[0], out_chns=fea[-1], act=act, norm=norm, bias=bias, dropout=dropout, upsample=upsample, halves=False)
        
        # first 2 layer
        feat_shape_0 = [patch_size[0], patch_size[1], patch_size[2]]
        feat_shape_1 = [patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]
        feat_shape_2 = [patch_size[0] // 4, patch_size[1] // 4, patch_size[2] // 4]

        self.layer2 = LinearInceptioConvnAtnnBlock(
            dim=fea[1], num_heads=attn_dict['head_dims'][0],
            Inception_window_size=[[4, 16, 16]],
            qk_dim_compresion=attn_dict['qk_dim_compresion'][0],
            qk_nonlin=attn_dict['qk_nonlin'],
            feat_shape=feat_shape_1,
            debug=attn_dict.get('debug', False)
        )


        self.final_conv = Conv["conv", spatial_dims](fea[-1], out_channels, kernel_size=1)

    def forward(self, x0, x1, x2):

        u2 = self.upcat_2(x2, x1)

        with torch.cuda.amp.autocast(enabled=False):
            u2 = u2.to(dtype=torch.float32)
            u2 = self.layer2(u2)

        u1 = self.upcat_1(u2, x0)
        logits = self.final_conv(u1)
        return logits
    


class ConvPixelInceptionEncoder_2Layer(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        patch_size: Sequence[int] = (20, 256, 224),
        features: Sequence[int] = (32, 32, 64, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        # now we add attention related arguments
        attn_dict: dict = None,
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 4)
        self.pos_enc = attn_dict['pos_enc']
        print("=======>", attn_dict)
        print("positional encoding", self.pos_enc)
        if self.pos_enc == 1:
            self.pos_embed = PositionalEncoding1D(fea[0])
        elif self.pos_enc == 2:
            self.pos_embed = PositionalEncoding3D(fea[0])

        # patch embedding layer but is pixel embedding
        # 6 layer encoder 6 layer decoder for small model
        # 2 layer for input shape
        # 2 layer for 1/2 shape
        # 2 layer for 1/4 shape

        self.input_norm = nn.InstanceNorm3d(in_channels, affine=True)
        self.patch_embed = nn.Conv3d(in_channels, fea[0], kernel_size=3, stride=1, padding=1)

        # TwoConv(spatial_dims=3, in_chns=in_channels, out_chns=fea[0], act=act, norm=norm, bias=bias, dropout=dropout)

        # nn.Conv3d(in_channels, fea[0], kernel_size=3, stride=1, padding=1) # 1x1x1 conv, only change channel

        # first 2 layer
        feat_shape_0 = [patch_size[0], patch_size[1], patch_size[2]]
        feat_shape_1 = [patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]
        feat_shape_2 = [patch_size[0] // 4, patch_size[1] // 4, patch_size[2] // 4]

        Inception_window_size = attn_dict['Inception_window_size']
        # self.layer_0 = LinearInceptionAtnnBlock(
        #     dim=fea[0], num_heads=attn_dict['head_dims'][0], 
        #     feat_shape=feat_shape_0,
        #     Inception_window_size=[[8, 32, 32]],
        #     qk_dim_compresion=attn_dict['qk_dim_compresion'][0],
        #     qk_nonlin=attn_dict['qk_nonlin'],
        # )
        # downsampling to feat_shape_1
        self.down_0 = torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        # torch.nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        # 
        self.conv_0 = nn.Conv3d(fea[0], fea[1], kernel_size=1, stride=1, padding=0)

        self.layer_1 = nn.ModuleList(
            [
               LinearInceptionAtnnBlock(
                    dim=fea[1], num_heads=attn_dict['head_dims'][0], 
                    feat_shape=feat_shape_1,
                    Inception_window_size=Inception_window_size[0], # 8x112x112
                    qk_dim_compresion=attn_dict['qk_dim_compresion'][0],
                    qk_nonlin=attn_dict['qk_nonlin'],
                    debug=attn_dict.get('debug', False)
                ) for i in range(attn_dict['stack_num'][0])

            ]
        )

        # downsampling to feat_shape_2
        self.down_1 = torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        # torch.nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        # torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv_1 = nn.Conv3d(fea[1], fea[2], kernel_size=1, stride=1, padding=0)
        self.layer_2 = nn.ModuleList(
            [
                LinearInceptionAtnnBlock(
                      dim=fea[2], num_heads=attn_dict['head_dims'][1], 
                      feat_shape=feat_shape_2,
                      Inception_window_size=Inception_window_size[1], # 4x56x56
                      qk_dim_compresion=attn_dict['qk_dim_compresion'][1],
                      qk_nonlin=attn_dict['qk_nonlin'],
                      debug=attn_dict.get('debug', False),
                 ) for i in range(attn_dict['stack_num'][1])
    
                ]
        )

        self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight.view(m.weight.size(0), -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # x: B, C, H, W, D
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x_input = x
            x = x.to(dtype=torch.float32)
            # x = self.input_norm(x)
            x = self.patch_embed(x)
            
            if torch.isnan(x).any():
                print("x", x.shape, x.min(), x.max(), x.dtype)
                # check x_input
                print("x_input", x_input.shape, x_input.min(), x_input.max(), x_input.dtype)
                # check if weight in patch_embed has NaN
                print("patch_embed", self.patch_embed.weight.shape, self.patch_embed.weight.min(), self.patch_embed.weight.max(), self.patch_embed.weight.dtype)
                raise ValueError("x after patch embd has NaN")
            
            # x = self.layer_0(x)
            x0 = x
            x = self.down_0(x)
            x = self.conv_0(x)

            if torch.isnan(x).any():
                print("x", x.shape, x.min(), x.max(), x.dtype)
                raise ValueError("x after conv_0 has NaN")
            
            B, C, H, W, D = x.shape
            if self.pos_enc == 1:
                x = x.view(B,C,-1).permute(0,2,1)
                pos_embed = self.pos_embed(x)
                x = x + pos_embed
                x = x.permute(0,2,1).view(B,C,H,W,D)
            elif self.pos_enc == 2:
                pos_embed = self.pos_embed(x)
                pos_embed.requires_grad = False
                x = x + pos_embed

            
            x = x.to(dtype=torch.float32)
            for blk in self.layer_1:
                x = blk(x)
            x1 = x
            x = self.down_1(x)
            x = self.conv_1(x)
            
            for blk in self.layer_2:
                x = blk(x)
            x2 = x

        # print("x0", x0.shape, x0.min(), x0.max(), x0.dtype)
        # print("x1", x1.shape, x1.min(), x1.max(), x1.dtype)
        # print("x2", x2.shape, x2.min(), x2.max(), x2.dtype)

        # raise ValueError("stop here")

        return x0, x1, x2

class ConvPixelInceptionConvEncoder_2Layer(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        patch_size: Sequence[int] = (20, 256, 224),
        features: Sequence[int] = (32, 32, 64, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        # now we add attention related arguments
        attn_dict: dict = None,
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 4)
        self.pos_enc = attn_dict['pos_enc']
        print("=======>", attn_dict)
        print("positional encoding", self.pos_enc)
        if self.pos_enc == 1:
            self.pos_embed = PositionalEncoding1D(fea[0])
        elif self.pos_enc == 2:
            self.pos_embed = PositionalEncoding3D(fea[0])

        # patch embedding layer but is pixel embedding
        # 6 layer encoder 6 layer decoder for small model
        # 2 layer for input shape
        # 2 layer for 1/2 shape
        # 2 layer for 1/4 shape

        self.patch_embed = nn.Sequential(
            nn.Conv3d(in_channels, fea[0], kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(1, fea[0], eps=1e-6),
            nn.InstanceNorm3d(fea[0], affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(fea[0], fea[0], kernel_size=3, stride=1, padding=1),
        )

        # self.embed_feat_norm = nn.LayerNorm(fea[0])
        # self.input_norm = nn.InstanceNorm3d(in_channels, affine=True)
        # first 2 layer
        feat_shape_0 = [patch_size[0], patch_size[1], patch_size[2]]
        feat_shape_1 = [patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]
        feat_shape_2 = [patch_size[0] // 4, patch_size[1] // 4, patch_size[2] // 4]

        Inception_window_size = attn_dict['Inception_window_size']
        # self.layer_0 = LinearInceptionAtnnBlock(
        #     dim=fea[0], num_heads=attn_dict['head_dims'][0], 
        #     feat_shape=feat_shape_0,
        #     Inception_window_size=[[8, 32, 32]],
        #     qk_dim_compresion=attn_dict['qk_dim_compresion'][0],
        #     qk_nonlin=attn_dict['qk_nonlin'],
        # )
        # downsampling to feat_shape_1
        self.down_0 = torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        # torch.nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        # torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv_0 = nn.Conv3d(fea[0], fea[1], kernel_size=3, stride=1, padding=1)

        if attn_dict['stack_num'][0] > 0:
            self.layer_1 = nn.ModuleList(
                [
                LinearInceptioConvnAtnnBlock(
                        dim=fea[1], num_heads=attn_dict['head_dims'][0], 
                        feat_shape=feat_shape_1,
                        Inception_window_size=Inception_window_size[0], # 8x112x112
                        qk_dim_compresion=attn_dict['qk_dim_compresion'][0],
                        qk_nonlin=attn_dict['qk_nonlin'],
                        debug=attn_dict.get('debug', False)
                    ) for i in range(attn_dict['stack_num'][0])

                ]
            )
        else:
            self.layer_1 =nn.ModuleList([
                InvertedResidual(in_dim=fea[1], 
                kernel_size=3
            )])

        # downsampling to feat_shape_2
        self.down_1 = torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        # torch.nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        # torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv_1 = nn.Conv3d(fea[1], fea[2], kernel_size=3, stride=1, padding=1)
        if attn_dict['stack_num'][1] > 0: 
            self.layer_2 = nn.ModuleList(
            [
                LinearInceptioConvnAtnnBlock(
                      dim=fea[2], num_heads=attn_dict['head_dims'][1], 
                      feat_shape=feat_shape_2,
                      Inception_window_size=Inception_window_size[1], # 4x56x56
                      qk_dim_compresion=attn_dict['qk_dim_compresion'][1],
                      qk_nonlin=attn_dict['qk_nonlin'],
                      debug=attn_dict.get('debug', False),
                 ) for i in range(attn_dict['stack_num'][1])
    
                ]
        )
        else:
            self.layer_2 =nn.ModuleList([
                InvertedResidual(in_dim=fea[2], 
                kernel_size=3
            )])

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight.view(m.weight.size(0), -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # x: B, C, H, W, D
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            # x = self.input_norm(x)
            x_input = x
            x = x.to(dtype=torch.float32)
            x = self.patch_embed(x)
            # print("x embed", x.shape)
            # x embed torch.Size([2, 32, 16, 224, 224])
            # x down 0 torch.Size([2, 32, 8, 112, 112])
            # x down 1 torch.Size([2, 64, 4, 56, 56])
            # x = self.layer_0(x)
            if torch.isnan(x).any():
                print("x", x.shape, x.min(), x.max(), x.dtype)
                # check x_input
                print("x_input", x_input.shape, x_input.min(), x_input.max(), x_input.dtype)
                # check if weight in patch_embed has NaN
                print("patch_embed", self.patch_embed.weight.shape, self.patch_embed.weight.min(), self.patch_embed.weight.max(), self.patch_embed.weight.dtype)
                raise ValueError("x after patch embd has NaN")
            
            x0 = x
            x = self.down_0(x)

            # print("x down 0", x.shape)

            if torch.isnan(x).any():
                print("x", x.shape, x.min(), x.max(), x.dtype)
                raise ValueError("x after down_0 has NaN")
            
            x = self.conv_0(x)

            if torch.isnan(x).any():
                print("x", x.shape, x.min(), x.max(), x.dtype)
                raise ValueError("x after conv_0 has NaN")
            

            B, C, H, W, D = x.shape
            if self.pos_enc == 1:
                x = x.view(B,C,-1).permute(0,2,1)
                pos_embed = self.pos_embed(x)
                x = x + pos_embed
                x = x.permute(0,2,1).view(B,C,H,W,D)
            elif self.pos_enc == 2:
                pos_embed = self.pos_embed(x)
                pos_embed.requires_grad = False
                x = x + pos_embed
            
            if torch.isnan(x).any():
                print("x", x.shape, x.min(), x.max(), x.dtype)
                raise ValueError("x after pos_embed has NaN")

            

            for layer_i, blk in enumerate(self.layer_1):
                if torch.isnan(x).any():
                    print("layer 1 x", layer_i,  x.shape, x.min(), x.max(), x.dtype)
                    raise ValueError("x has NaN")
                x = blk(x)
            x1 = x
            x = self.down_1(x)
            x = self.conv_1(x)

            # print("x down 1", x.shape)
            
            for layer_i, blk in enumerate(self.layer_2):
                if torch.isnan(x).any():
                    print("x", x.shape, x.min(), x.max(), x.dtype)
                    print("layer 2 x", layer_i,  x.shape, x.min(), x.max(), x.dtype)
                    raise ValueError("x has NaN")
                
                x = blk(x)
            x2 = x

        return x0, x1, x2


class LinearInceptionConvAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
        use_lora=0,
        qk_nonlin='softmax',
        Inception_window_size=[[4, 4, 4]],
        qk_dim_compresion=1,
        debug=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qk_nonlin = qk_nonlin
        self.Inception_window_size = Inception_window_size

        qk_dim = dim // qk_dim_compresion
        v_dim = dim

        self.qk_dim = qk_dim
        self.v_dim = v_dim

        self.q = nn.Conv3d(dim, qk_dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.k = nn.Conv3d(dim, qk_dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.v = nn.Conv3d(dim, v_dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.proj = nn.Conv3d(v_dim, v_dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

        self.input_size = input_size
        assert input_size[1] == input_size[2]
        self.max_window = [max([i[0] for i in Inception_window_size]),
                            max([i[1] for i in Inception_window_size]), 
                            max([i[2] for i in Inception_window_size])]
        
        self.debug = debug

    def forward(self, x):
        # input x: B, C, H, W, D
        B, C, H, W, D = x.shape
        assert H == self.input_size[0]
        assert W == self.input_size[1]
        assert D == self.input_size[2]

        assert C == self.v_dim
        # building q, k, v
        x_3d = x 

        # debug NaN
        if torch.isnan(x_3d).any():
            print("x_3d", x_3d.shape, x_3d.min(), x_3d.max(), x_3d.dtype)
            raise ValueError("x_3d has NaN")

        q_3d = self.q(x)
        k_3d = self.k(x)
        v_3d = self.v(x)


        if torch.isnan(q_3d).any():
            print("q_3d", q_3d.shape, q_3d.min(), q_3d.max(), q_3d.dtype)
            raise ValueError("q_3d has NaN")
        if torch.isnan(k_3d).any():
            print("k_3d", k_3d.shape, k_3d.min(), k_3d.max(), k_3d.dtype)
            raise ValueError("k_3d has NaN")
        if torch.isnan(v_3d).any():
            print("v_3d", v_3d.shape, v_3d.min(), v_3d.max(), v_3d.dtype)
            raise ValueError("v_3d has NaN")
        

        # linear attention idea here
        if self.qk_nonlin == 'elu':
            q_3d = torch.nn.functional.elu(q_3d) + 1 + 1e-6
            k_3d = torch.nn.functional.elu(k_3d) + 1 + 1e-6

        elif self.qk_nonlin == 'softmax':
            q_3d = q_3d.reshape(B, C, -1)
            q_3d = q_3d.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
            q_3d = q_3d.softmax(dim=-1) + 1e-6

            k_3d = k_3d.reshape(B, C, -1)
            k_3d = k_3d.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
            k_3d = k_3d.softmax(dim=1) + 1e-6 # softmax along the sequence length

            # q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
            # q = q.softmax(dim=-1)
            # q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
            # k = k.softmax(dim=1)
        else:
            raise NotImplementedError

        q_3d = q_3d.view(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
        k_3d = k_3d.view(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
        v_3d = v_3d.view(B*self.num_heads, self.v_dim // self.num_heads, H, W, D)

        if self.debug:
            # do global linear atteention, no inception idea
            q_3d = q_3d.reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)
            k_3d = k_3d.reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)
            v_3d = v_3d.reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)
      
            kv_3d = (k_3d.transpose(-2, -1) @ v_3d)  # B, num_heads, F, F
            k_3d = k_3d.sum(dim=-2, keepdim=True) # B, num_heads, 1, F
            attn = q_3d @ kv_3d * self.scale # B, num_heads, H*W*D, F
            # print("attn", attn.shape, attn.min(), attn.max(), attn.dtype)
            # print("q_3d", q_3d.shape, q_3d.min(), q_3d.max(), q_3d.dtype)
            # print("k_3d", k_3d.shape, k_3d.min(), k_3d.max(), k_3d.dtype)

            # print("attn results", attn.shape, attn.min(), attn.max(), attn.dtype)

            Z = q_3d @ k_3d.transpose(-2, -1) + 1e-6 # B, num_heads, H*W*D, F @ B, num_heads, F, 1 = B, num_heads, H*W*D, 1

            # print("normalization", Z.shape, Z.min(), Z.max(), Z.dtype)

            attn = attn / Z

            # print("normalization results", attn.shape, attn.min(), attn.max(), attn.dtype)

            attn = attn.permute(0, 1, 3, 2).reshape(B, C, H, W, D)

            output_attn = self.proj(attn)
            output_attn = self.proj_drop(output_attn)
            # print("output_attn", output_attn.shape, output_attn.min(), output_attn.max(), output_attn.dtype)
            return output_attn
        
        else:
            intergral_kv_map = build_3D_kv_integral_image(k_3d, v_3d) # B, F, F, H, W, D
            intergral_k_map = build_3D_k_integral_image(k_3d) # B, F, H, W, D

            intergral_kv_map_reshape = intergral_kv_map.reshape(B*self.num_heads, self.qk_dim // self.num_heads, self.v_dim // self.num_heads, H, W, D) # pad new shape need to get new shape
            intergral_kv_map_reshape = intergral_kv_map_reshape.reshape(B*self.num_heads*(self.qk_dim // self.num_heads) * ( self.v_dim // self.num_heads), 1, H, W, D) # B, F, F, -1


            intergral_k_map_reshape = intergral_k_map.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
            intergral_k_map_reshape = intergral_k_map_reshape.reshape(B*self.num_heads*(self.qk_dim // self.num_heads), 1, H, W, D) # B, F, F, -1

            output_attn = torch.zeros_like(x_3d).to(dtype=torch.float32).to(x.device)
            for window_size in self.Inception_window_size:
                assert len(window_size) == 3
                kernel, dialation_size = build_convolve_kernel_3D_from_window_size(window_size, device=x.device)
                kernel = kernel.unsqueeze(0).unsqueeze(0)
                # make kernel not trainable
                kernel.requires_grad = False

                # print("kernel", kernel.shape, kernel.min(), kernel.max(), kernel.dtype)
                # print("dialation_size", dialation_size)
                # print("intergral_kv_map_reshape", intergral_kv_map_reshape.shape, intergral_kv_map_reshape.min(), intergral_kv_map_reshape.max(), intergral_kv_map_reshape.dtype)
                # print("intergral_k_map_reshape", intergral_k_map_reshape.shape, intergral_k_map_reshape.min(), intergral_k_map_reshape.max(), intergral_k_map_reshape.dtype)

                dialation_size = dialation_size - 1
        
                window_kv = torch.nn.functional.conv3d(input=intergral_kv_map_reshape, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1)
                window_k = torch.nn.functional.conv3d(input=intergral_k_map_reshape, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1) 

                target_shape = torch.tensor([H,W,D])
                window_kv_shape = torch.tensor(window_kv.shape[-3:])
                pad_h = (target_shape[0] - window_kv_shape[0]) // 2
                pad_h_end = target_shape[0] - window_kv_shape[0] - pad_h
                pad_w = (target_shape[1] - window_kv_shape[1]) // 2
                pad_w_end = target_shape[1] - window_kv_shape[1] - pad_w
                pad_d = (target_shape[2] - window_kv_shape[2]) // 2
                pad_d_end = target_shape[2] - window_kv_shape[2] - pad_d
            
                pad_operation = nn.ReplicationPad3d((pad_d, pad_d_end, pad_w, pad_w_end, pad_h, pad_h_end))

                window_kv_pad = pad_operation(window_kv)
                window_k_pad = pad_operation(window_k)

                window_kv_pad = window_kv_pad.view(B*self.num_heads, self.qk_dim // self.num_heads, self.v_dim // self.num_heads, H, W, D) # B, F, F, -1
                window_k_pad = window_k_pad.view(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D) # B, F, -1

                
                attn = torch.einsum('bhwdf,bhwdfc->bhwdc',q_3d.permute(0,2,3,4,1), window_kv_pad.permute(0,3,4,5,1,2)) * self.scale
                # (B*h, N, Fqk) (B*h, N, Fqk, Fv)
                # print("attn", attn.shape, attn.min(), attn.max(), attn.dtype)
                norm = (q_3d.permute(0,2,3,4,1) * window_k_pad.permute(0,2,3,4,1)).sum(-1, keepdim=True) + 1e-6

                # print("norm", norm.shape, norm.min(), norm.max(), norm.dtype)

                attn = attn / norm # (B*h, H,W,D, C)
                V = attn.permute(0, 4, 1, 2, 3).reshape(B, self.v_dim, H, W, D)
                output_attn = output_attn + V
                
            output_attn = self.proj(output_attn)
            output_attn = self.proj_drop(output_attn)
            return output_attn


class LinearInceptioConvnAtnnBlock(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        feat_shape,
        Inception_window_size,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.SiLU,
        norm_layer=nn.GroupNorm,
        attn_func=LinearInceptionConvAttention,
        qk_dim_compresion=1.0,
        qk_nonlin='softmax',
        mlp_ratio=4.0,
        debug=False,
    ):
        super().__init__()


        self.norm1 = nn.InstanceNorm3d(dim, affine=True)
        # norm_layer(1, dim, eps=1e-6)
        self.attn = attn_func(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            input_size=feat_shape,
            use_lora=0,
            qk_nonlin=qk_nonlin,
            Inception_window_size=Inception_window_size,
            debug=debug,
        )
        self.norm2 = nn.InstanceNorm3d(dim, affine=True)
        # norm_layer(1,dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.irb = InvertedResidual(dim, hidden_dim=int(dim * mlp_ratio), act_layer=act_layer, drop=drop,
                                    kernel_size=3)

    def forward(self, x):        
        # input x: B, C, H, W, D
        B, C, H, W, D = x.shape
        # norm 
        if torch.isnan(x).any():
            print("x", x.shape, x.min(), x.max(), x.dtype)
            raise ValueError("x has NaN")
        
        x = self.norm1(x)
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.irb(x))
        return x



class InvertedResidual(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, kernel_size=3,
                 drop=0., act_layer=nn.SiLU):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        pad = (kernel_size - 1) // 2

        self.conv1 = nn.Sequential(
            nn.GroupNorm(1, in_dim, eps=1e-6),
            nn.Conv3d(in_dim, hidden_dim, 1, bias=False),
            act_layer(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size, padding=pad, groups=hidden_dim, bias=False),
            act_layer(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(hidden_dim, out_dim, 1, bias=False),
            nn.GroupNorm(1, out_dim, eps=1e-6)
        )
        self.drop = nn.Dropout3d(drop, inplace=True)

    def forward(self, x):
        # print("InvertedResidual x", x.shape) ([2, 32, 8, 112, 112])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)

        return x

class LinearInceptionAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
        use_lora=0,
        qk_nonlin='softmax',
        Inception_window_size=[[4, 4, 4]],
        debug=False,
        qk_dim_compresion=1,
    ):
        super().__init__()
        self.debug = debug
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.qk_dim = dim // qk_dim_compresion
        self.v_dim = dim

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qk_nonlin = qk_nonlin
        self.Inception_window_size = Inception_window_size

        if use_lora == 0:
            self.q = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
            self.k = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
            self.v = nn.Linear(dim, self.v_dim, bias=qkv_bias)
            assert attn_drop == 0.0  # do not use
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        elif use_lora == 1:
            self.q = lora.Linear(dim, self.qk_dim, r=16, lora_alpha=16)
            self.k = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
            self.v = lora.Linear(dim, self.v_dim, r=16, lora_alpha=16)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        elif use_lora == 2:
            self.q = lora.Linear(dim, self.qk_dim, r=16, lora_alpha=16)
            self.k = lora.Linear(dim, self.qk_dim, r=16, lora_alpha=16)
            self.v = lora.Linear(dim, dim, r=16, lora_alpha=16)
            self.proj = lora.Linear(dim, dim, r=16, lora_alpha=16)
            self.proj_drop = nn.Dropout(proj_drop)

        self.input_size = input_size
        assert input_size[1] == input_size[2]
        self.max_window = [max([i[0] for i in Inception_window_size]),
                            max([i[1] for i in Inception_window_size]), 
                            max([i[2] for i in Inception_window_size])]

    def forward(self, x):
        # input x: B, C, H, W, D
        B, C, H, W, D = x.shape
        assert H == self.input_size[0]
        assert W == self.input_size[1]
        assert D == self.input_size[2]

        # building q, k, v
        x_3d = x 
        x_token = x.view(B, C, -1).permute(0, 2, 1) # B, N, C

        B, N, C = x_token.shape
        # first we assume C is the same, later we change to C_qk and C_v
        q = (
            self.q(x_token)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x_token)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x_token)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # linear attention idea here
        if self.qk_nonlin == 'elu':
            q = torch.nn.functional.elu(q) + 1 + 1e-6
            k = torch.nn.functional.elu(k) + 1 + 1e-6

        elif self.qk_nonlin == 'softmax':
            q = q.softmax(dim=-1)
            k = k.softmax(dim=1) # softmax along the sequence length

            # q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
            # q = q.softmax(dim=-1)
            # q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
            # k = k.softmax(dim=1)
        else:
            raise NotImplementedError

        q_3d = q.reshape(B*self.num_heads, N, C // self.num_heads).permute(0, 2, 1).reshape(B*self.num_heads, C // self.num_heads, H, W, D)
        k_3d = k.reshape(B*self.num_heads, N, C // self.num_heads).permute(0, 2, 1).reshape(B*self.num_heads, C // self.num_heads, H, W, D)
        v_3d = v.reshape(B*self.num_heads, N, C // self.num_heads).permute(0, 2, 1).reshape(B*self.num_heads, C // self.num_heads, H, W, D)


        if self.debug:
            # do global linear atteention
            # print("attn", attn.shape, attn.min(), attn.max(), attn.dtype)
            # print("q_3d", q_3d.shape, q_3d.min(), q_3d.max(), q_3d.dtype)
            # print("k_3d", k_3d.shape, k_3d.min(), k_3d.max(), k_3d.dtype)

            # print("attn results", attn.shape, attn.min(), attn.max(), attn.dtype)


            q_3d = q_3d.reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)
            k_3d = k_3d.reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)
            v_3d = v_3d.reshape(B, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)
      
            kv_3d = (k_3d.transpose(-2, -1) @ v_3d)  # B, num_heads, F, F
            k_3d = k_3d.sum(dim=-2, keepdim=True) # B, num_heads, 1, F
            attn = q_3d @ kv_3d * self.scale # B, num_heads, H*W*D, F
            Z = q_3d @ k_3d.transpose(-2, -1)+ 1e-6 # B, num_heads, H*W*D, F @ B, num_heads, F, 1 = B, num_heads, H*W*D, 1
            attn = attn / Z

            output_attn = attn.permute(0, 1, 3, 2).reshape(B, C, H, W, D)

            output_attn = output_attn.view(B, C, -1).permute(0, 2, 1) # B, N, C
            output_attn = self.proj(output_attn)
            output_attn = self.proj_drop(output_attn)
            output_attn = output_attn.permute(0, 2, 1).view(B, C, H, W, D)

            # print("output_attn", output_attn.shape, output_attn.min(), output_attn.max(), output_attn.dtype)
            return output_attn
        
        else:
            intergral_kv_map = build_3D_kv_integral_image(k_3d, v_3d) # B, F, F, H, W, D
            intergral_k_map = build_3D_k_integral_image(k_3d) # B, F, H, W, D

            intergral_kv_map_reshape = intergral_kv_map.reshape(B*self.num_heads, self.qk_dim // self.num_heads, self.v_dim // self.num_heads, H, W, D) # pad new shape need to get new shape
            intergral_kv_map_reshape = intergral_kv_map_reshape.reshape(B*self.num_heads*(self.qk_dim // self.num_heads) * ( self.v_dim // self.num_heads), 1, H, W, D) # B, F, F, -1


            intergral_k_map_reshape = intergral_k_map.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
            intergral_k_map_reshape = intergral_k_map_reshape.reshape(B*self.num_heads*(self.qk_dim // self.num_heads), 1, H, W, D) # B, F, F, -1
            
            output_attn = torch.zeros_like(x_3d).to(dtype=torch.float32).to(x.device)
            for window_size in self.Inception_window_size:
                assert len(window_size) == 3
                kernel, dialation_size = build_convolve_kernel_3D_from_window_size(window_size, device=x.device)
                kernel = kernel.unsqueeze(0).unsqueeze(0)
                # make kernel not trainable
                kernel.requires_grad = False

                # print("kernel", kernel.shape, kernel.min(), kernel.max(), kernel.dtype)
                # print("dialation_size", dialation_size)
                # print("intergral_kv_map_reshape", intergral_kv_map_reshape.shape, intergral_kv_map_reshape.min(), intergral_kv_map_reshape.max(), intergral_kv_map_reshape.dtype)
                # print("intergral_k_map_reshape", intergral_k_map_reshape.shape, intergral_k_map_reshape.min(), intergral_k_map_reshape.max(), intergral_k_map_reshape.dtype)

                dialation_size = dialation_size - 1
        
                window_kv = torch.nn.functional.conv3d(input=intergral_kv_map_reshape, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1)
                window_k = torch.nn.functional.conv3d(input=intergral_k_map_reshape, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1) 

                target_shape = torch.tensor([H,W,D])
                window_kv_shape = torch.tensor(window_kv.shape[-3:])
                pad_h = (target_shape[0] - window_kv_shape[0]) // 2
                pad_h_end = target_shape[0] - window_kv_shape[0] - pad_h
                pad_w = (target_shape[1] - window_kv_shape[1]) // 2
                pad_w_end = target_shape[1] - window_kv_shape[1] - pad_w
                pad_d = (target_shape[2] - window_kv_shape[2]) // 2
                pad_d_end = target_shape[2] - window_kv_shape[2] - pad_d
            
                pad_operation = nn.ReplicationPad3d((pad_d, pad_d_end, pad_w, pad_w_end, pad_h, pad_h_end))

                window_kv_pad = pad_operation(window_kv)
                window_k_pad = pad_operation(window_k)

                window_kv_pad = window_kv_pad.view(B*self.num_heads, self.qk_dim // self.num_heads, self.v_dim // self.num_heads, H, W, D) # B, F, F, -1
                window_k_pad = window_k_pad.view(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D) # B, F, -1

                
                attn = torch.einsum('bhwdf,bhwdfc->bhwdc',q_3d.permute(0,2,3,4,1), window_kv_pad.permute(0,3,4,5,1,2)) * self.scale
                # (B*h, N, Fqk) (B*h, N, Fqk, Fv)
                # print("attn", attn.shape, attn.min(), attn.max(), attn.dtype)
                norm = (q_3d.permute(0,2,3,4,1) * window_k_pad.permute(0,2,3,4,1)).sum(-1, keepdim=True) + 1e-6

                # print("norm", norm.shape, norm.min(), norm.max(), norm.dtype)

                attn = attn / norm # (B*h, H,W,D, C)
                V = attn.permute(0, 4, 1, 2, 3).reshape(B, self.v_dim, H, W, D)
                output_attn = output_attn + V
                

            output_attn = output_attn.view(B, C, -1).permute(0, 2, 1) # B, N, C
            output_attn = self.proj(output_attn)
            output_attn = self.proj_drop(output_attn)
            output_attn = output_attn.permute(0, 2, 1).view(B, C, H, W, D)
            return output_attn

class LinearInceptionAtnnBlock(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        feat_shape,
        Inception_window_size,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=LinearInceptionAttention,
        qk_dim_compresion=1.0,
        qk_nonlin='softmax',
        mlp_ratio=4.0,
        debug=False,
    ):
        super().__init__()


        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            input_size=feat_shape,
            use_lora=0,
            qk_nonlin=qk_nonlin,
            Inception_window_size=Inception_window_size,
            debug=debug,
            qk_dim_compresion=qk_dim_compresion,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x):        
        # input x: B, C, H, W, D
        B, C, H, W, D = x.shape
        # norm 
        x_token = x.view(B, C, -1).permute(0, 2, 1) # B, N, C
        x_token = self.norm1(x_token)
        x_3d = x_token.permute(0, 2, 1).view(B, C, H, W, D)
        x_3d =  x_3d + self.drop_path(self.attn(x_3d))
        x_token = x_3d.view(B, C, -1).permute(0, 2, 1) # B, N, C
        x_token = self.norm2(x_token)
        x_token = x_token + self.drop_path(self.mlp(x_token))
        x_3d = x_token.permute(0, 2, 1).view(B, C, H, W, D)
 
        return x_3d



def create_3d_grid_via_numpy(D, H, W, abc):
    # Compute the starting indices for each grid along each dimension
    a, b, c = abc
    z1_coords = np.arange(0, D, a)
    y1_coords = np.arange(0, H, b)
    x1_coords = np.arange(0, W, c)

    # Add boundary conditions
    if z1_coords[-1] + a < D:
        z1_coords = np.append(z1_coords, D - a)
    if y1_coords[-1] + b < H:
        y1_coords = np.append(y1_coords, H - b)
    if x1_coords[-1] + c < W:
        x1_coords = np.append(x1_coords, W - c)

    # Compute the corresponding ending indices
    z2_coords = np.minimum(z1_coords + a - 1, D - 1)
    y2_coords = np.minimum(y1_coords + b - 1, H - 1)
    x2_coords = np.minimum(x1_coords + c - 1, W - 1)

    # Create all combinations of start and end indices
    z1, y1, x1 = np.meshgrid(z1_coords, y1_coords, x1_coords, indexing='ij')
    z2, y2, x2 = np.meshgrid(z2_coords, y2_coords, x2_coords, indexing='ij')

    # Stack the coordinates to create grid regions
    start_coords = np.stack([z1, y1, x1], axis=-1).reshape(-1, 3)
    end_coords = np.stack([z2, y2, x2], axis=-1).reshape(-1, 3)

    # Combine start and end coordinates into grid tuples
    # grid = [(tuple(start), tuple(end)) for start, end in zip(start_coords, end_coords)]
    # grid = np.array(grid)
    grid = np.concatenate([start_coords, end_coords], axis=1)

    return grid


def retrieval_KV_3D_integral_image(integral_image, x1, y1, z1, x2, y2, z2):
    # integral_image: B, F, F, D+1, H+1, W+1
    # retrieval_loc: Nx6, 6 for x1, y1, z1, x2, y2, z2
    # output: B x F x F x (x2 - x1 + 1) x (y2 - y1 + 1) x (z2 - z1 + 1)
    B, F, F, D_, H_, W_ = integral_image.shape
    subvolume = integral_image[:,:,:, x2+1, y2+1, z2+1] - integral_image[:,:,:, x1, y2+1, z2+1] - integral_image[:,:,:, x2+1, y1, z2+1] - integral_image[:,:,:, x2+1, y2+1, z1] + integral_image[:,:,:, x1, y1, z2+1] + integral_image[:,:,:, x1, y2+1, z1] + integral_image[:,:,:, x2+1, y1, z1] - integral_image[:,:,:, x1, y1, z1]
 
    return subvolume

def retrieval_Qs_window(q, regions):
    # q: B, F, D, H, W
    # regions: Nx6, 6 for x1, y1, z1, x2, y2, z2
    # output: B x F x N x (x2 - x1 + 1) x (y2 - y1 + 1) x (z2 - z1 + 1)

    B, F, D, H, W = q.shape
    window_d = regions[:, 3] - regions[:, 0] + 1
    assert torch.all(window_d - window_d[0] == 0), "window size should be the same"

    window_h = regions[:, 4] - regions[:, 1] + 1
    assert torch.all(window_h - window_h[0] == 0), "window size should be the same"

    window_w = regions[:, 5] - regions[:, 2] + 1
    assert torch.all(window_w - window_w[0] == 0), "window size should be the same"
    
    x1, y1, z1, x2, y2, z2 = regions[:, 0], regions[:, 1], regions[:, 2], regions[:, 3], regions[:, 4], regions[:, 5]
    q_retrieval = torch.zeros(B, F, regions.shape[0],  window_d[0], window_h[0], window_w[0], device=q.device) 
    for i in range(regions.shape[0]):
        q_retrieval[:, :, i, :, :, :] = q[:, :, x1[i]:x2[i]+1, y1[i]:y2[i]+1, z1[i]:z2[i]+1]

    return q_retrieval


def retrieval_KVs_3D_integral_image(integral_image, regions):
    # integral_image: B, F, F, D+1, H+1, W+1
    # regions: Nx6, 6 for x1, y1, z1, x2, y2, z2
    # output: B x F x F x N
    B, F, F, D_, H_, W_ = integral_image.shape

    x1, y1, z1, x2, y2, z2 = regions[:, 0], regions[:, 1], regions[:, 2], regions[:, 3], regions[:, 4], regions[:, 5]


    assert torch.all(x2 > x1 ), "x1 should be less than x2"
    assert torch.all(y2 > y1), "y1 should be less than y2"
    assert torch.all(z2 > z1), "z1 should be less than z2"
    
    assert torch.all(x1 >= 0), "x1 should be greater than 0"
    assert torch.all(y1 >= 0), "y1 should be greater than 0"
    assert torch.all(z1 >= 0), "z1 should be greater than 0"

    assert torch.all(x2 < D_-1), "x2 should be less than W"
    assert torch.all(y2 < H_-1), "y2 should be less than H"
    assert torch.all(z2 < W_-1), "z2 should be less than D"

    # Retrieve subvolumes using advanced indexing
    subvolume = (
        integral_image[:, :, :, x2+1, y2+1, z2+1]
        - integral_image[:, :, :, x1, y2+1, z2+1]
        - integral_image[:, :, :, x2+1, y1, z2+1]
        - integral_image[:, :, :, x2+1, y2+1, z1]
        + integral_image[:, :, :, x1, y1, z2+1]
        + integral_image[:, :, :, x1, y2+1, z1]
        + integral_image[:, :, :, x2+1, y1, z1]
        - integral_image[:, :, :, x1, y1, z1]
    )
    # print("subvolume shape", subvolume.shape)

    return subvolume

def build_convolve_kernel_3D_from_window_size(window_size, device='cuda'):
    # window_size: dxhxw
    # output kernel: dxhxw
    window_size = torch.tensor(window_size)
    d, h, w = window_size
    kernel = torch.zeros(2,2,2, device=device)
    kernel[1, 1, 1] = 1
    kernel[1, 1, 0] = -1
    kernel[1, 0, 1] = -1
    kernel[1, 0, 0] = 1
    kernel[0, 1, 1] = -1
    kernel[0, 1, 0] = 1
    kernel[0, 0, 1] = 1
    kernel[0, 0, 0] = -1
    # calculate the dialation size for each dimension
    # assert d % 2 == 0, f"d should be even {d}"
    # assert h % 2 == 0, f"h should be even {h}"
    # assert w % 2 == 0, f"w should be even {w}"
    dialation_size = window_size 
    # kernel = torch.zeros(window_size, device=device)
    # kernel[d-1, h-1, w-1] = 1
    # kernel[d-1, h-1, 0] = -1
    # kernel[d-1, 0, w-1] = -1
    # kernel[d-1, 0, 0] = 1
    # kernel[0, h-1, w-1] = -1
    # kernel[0, h-1, 0] = 1
    # kernel[0, 0, w-1] = 1
    # kernel[0, 0, 0] = -1

    return kernel, dialation_size

def pad_intergral_kv_map(intergral_kv_map, max_window, window_size):
    _, _, H, W, D = intergral_kv_map.shape
    start_h = (max_window[0]  - window_size[0]) // 2
    start_w = (max_window[1]  - window_size[1]) // 2
    start_d = (max_window[2]  - window_size[2]) // 2
    end_h = max_window[0]  - window_size[0] - start_h
    end_w = max_window[1]  - window_size[1] - start_w
    end_d = max_window[2]  - window_size[2] - start_d


    intergral_kv_map_pad = intergral_kv_map[:, :, start_h : H - end_h, start_w : W - end_w, start_d : D - end_d]
    return intergral_kv_map_pad 


def build_3D_k_integral_image(k, pad=[0, 0, 0]):
    # k: B, F, H, W, D
    B, F, D, H, W = k.shape
    pad = torch.tensor(pad)
    if torch.all(pad==0):
        output = torch.cumsum(k, dim=2)
        output = torch.cumsum(output, dim=3)
        output = torch.cumsum(output, dim=4)
    else:
        half_pad_h = pad[0] // 2
        half_pad_w = pad[1] // 2
        half_pad_d = pad[2] // 2
        # pad the k v to be B, F, H+pad, W+pad, D+pad
        k_pad = torch.nn.functional.pad(k, (half_pad_d, half_pad_d, half_pad_w, half_pad_w, half_pad_h, half_pad_h), mode='constant', value=0)
        output = torch.cumsum(k_pad, dim=2)
        output = torch.cumsum(output, dim=3)
        output = torch.cumsum(output, dim=4)

        if not torch.all(torch.isinf(output) == 0):
            print("inf exist in output, replace to max value")
            min_value = output[torch.isinf(output) == 0].min()
            max_value = output[torch.isinf(output) == 0].max()
            output = output.clip(min=min_value, max=max_value)

        # how the code will be NaN?
        assert torch.all(torch.isnan(output) == 0), "output should not have NaN"
    
    return output

def build_3D_kv_integral_image(k, v, pad=[0, 0, 0]):
    # k: B, F, H, W, D
    # v: B, F, H, W, D
    # output: B, F, F, D+1, H+1, W+1
    pad = torch.tensor(pad)
    if torch.all(pad==0):
        assert k.shape == v.shape
        B, F, D, H, W = k.shape
        k_ = k.unsqueeze(2) # B, F, 1, H, W, D
        v_ = v.unsqueeze(1) # B, 1, F, H, W, D
        kv_product = torch.matmul(k_, v_) # B, F, F, H, W, D

        output = torch.cumsum(kv_product, dim=3)
        output = torch.cumsum(output, dim=4)
        output = torch.cumsum(output, dim=5)
    else:
        half_pad_h = pad[0] // 2
        half_pad_w = pad[1] // 2
        half_pad_d = pad[2] // 2
        # pad the k v to be B, F, H+pad, W+pad, D+pad
        assert torch.all(torch.isnan(k) == 0), "k should not have NaN"
        assert torch.all(torch.isnan(v) == 0), "v should not have NaN"

        k_pad = torch.nn.functional.pad(k, (half_pad_d, pad[2] - half_pad_d, half_pad_w, pad[1] - half_pad_w, half_pad_h, pad[0]- half_pad_h), mode='constant', value=0)
        v_pad = torch.nn.functional.pad(v, (half_pad_d, pad[2] - half_pad_d, half_pad_w, pad[1] - half_pad_w, half_pad_h, pad[0] - half_pad_h), mode='constant', value=0)
        k_ = k_pad.unsqueeze(2) # B, F, 1, H, W, D
        v_ = v_pad.unsqueeze(1) # B, 1, F, H, W, D

        kv_product = torch.matmul(k_, v_)  

        # assert torch.all(torch.isnan(kv_product) == 0), "kv_product should not have NaN"
        output = torch.cumsum(kv_product, dim=3)

        output = torch.cumsum(output, dim=4)

        output = torch.cumsum(output, dim=5)


        # if inf exist in output (too large value add up), replace the value to max value
    if not torch.all(torch.isinf(output) == 0):
        print("inf exist in output, replace to max value")
        min_value = output[torch.isinf(output) == 0].min()
        max_value = output[torch.isinf(output) == 0].max()
        output = output.clip(min=min_value, max=max_value)


    # assert no inf and Nan in output
    if not torch.all(torch.isnan(output) == 0):
        print("NaN exist in output")
        # print NaN indices
        print("k", torch.max(k), torch.min(k), k.shape)
        print("v", torch.max(v), torch.min(v), v.shape)

        print("k_pad", torch.max(k_pad), torch.min(k_pad), k_pad.shape)
        print("v_pad", torch.max(v_pad), torch.min(v_pad), v_pad.shape)

        print("kv_product", kv_product.min(), kv_product.max())
        NaN_indices = torch.where(torch.isnan(output))
        NaN_indices_np = [i.cpu().numpy() for i in NaN_indices]
        print("NaN indices", NaN_indices_np)
        raise ValueError("output should not have NaN")

        # assert torch.all(torch.isnan(output) == 0), "output should not have NaN"
        

        # raise ValueError("pad is not implemented yet")
    return output


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)





class UpCat_ConvPixelFormer(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        is_pad: bool = True,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the encoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.
            is_pad: whether to pad upsampling features to fit features from encoder. Defaults to True.

        """
        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = nn.Conv3d(cat_chns + up_chns, out_chns, kernel_size=3, stride=1, padding=1) # 1x1x1 conv, only change channel
        #TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)
        self.is_pad = is_pad

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            if self.is_pad:
                # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
                dimensions = len(x.shape) - 2
                sp = [0] * (dimensions * 2)
                for i in range(dimensions):
                    if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                        sp[i * 2 + 1] = 1
                x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x
    




# class LinearInceptionAttention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         Inception_window_size,
#         num_heads=8,
#         qkv_bias=False,
#         qk_scale=None,
#         attn_drop=0.0,
#         proj_drop=0.0,
#         input_size=(4, 14, 14),
#         pos_enc=0,
#         qk_dim_compresion=1.0,
#         qk_nonlin='softmax',
#         window_q = True,
#         feedforward='mlp', #mlp
#     ):
#         super().__init__()
#         assert dim % num_heads == 0, "dim should be divisible by num_heads"
#         self.num_heads = num_heads
#         assert qk_nonlin in ['softmax', 'sigmoid', 'max']
#         self.qk_nonlin = qk_nonlin
#         self.qk_dim = int(dim*qk_dim_compresion)
#         self.v_dim = dim
#         self.window_q = window_q
#         self.feedforward = feedforward
#         self.norm= LayerNorm3d(dim)

#         self.q = torch.nn.Conv3d(dim, self.qk_dim, 1, stride=1, padding=0, bias=False)
#         self.k = torch.nn.Conv3d(dim, self.qk_dim, 1, stride=1, padding=0, bias=False)
#         self.v = torch.nn.Conv3d(dim, self.v_dim, 1, stride=1, padding=0, bias=False)
#         # self.to_qkv = nn.Linear(dim, self.qk_dim + self.qk_dim + self.v_dim, bias=qkv_bias)

#         if feedforward == 'conv':
#             self.proj = nn.Conv3d(dim, dim, 1,stride=1, padding=0, bias=False)
#             self.proj_drop = nn.Dropout(proj_drop)
#         else:
#             self.proj = FeedForward(dim, dim, dropout=proj_drop)
    

#         self.pos_enc = pos_enc
#         self.input_size = input_size

#         self.pos_embed = PositionalEncoding3D(dim) if pos_enc == 2 else None

#         assert input_size[1] == input_size[2]

#         self.Inception_window_size = Inception_window_size
#         max_window_h = max([i[0] for i in Inception_window_size])
#         max_window_w = max([i[1] for i in Inception_window_size])
#         max_window_d = max([i[2] for i in Inception_window_size])
#         self.max_window = [max_window_h, max_window_w, max_window_d]

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#         wq = self.q.weight.data
#         ww = self.k.weight.data
#         wv = self.v.weight.data
#         # torch.nn.init.xavier_uniform_(wq.view([wq.shape[0], -1]))  
#         # torch.nn.init.xavier_uniform_(ww.view([ww.shape[0], -1]))
#         # torch.nn.init.xavier_uniform_(wv.view([wv.shape[0], -1]))
#         nn.init.trunc_normal_(wq, std=0.02)
#         nn.init.trunc_normal_(ww, std=0.02)
#         nn.init.trunc_normal_(wv, std=0.02)


#     def forward(self, x):
#         # when go to attention, layernnorm first
#         # check x data type
#         x = x.to(dtype=torch.float32)

#         x = self.norm(x)
#         t1 = time.time()
#         B, C, H, W, D = x.shape

#         if self.pos_enc == 2:
#             pos_embed = self.pos_embed(x)
#             x = x + pos_embed

#         output_attn = x.view(B*self.num_heads, self.v_dim // self.num_heads, H, W, D)
#         # x = x.view(B, C, -1)
#         # x = x.permute(0, 2, 1)
#         # qkv = self.to_qkv(x)
#         # q = qkv[:, :, :self.qk_dim].permute(0, 2, 1).reshape(B, self.qk_dim, H, W, D)
#         # k = qkv[:, :,self.qk_dim:2*self.qk_dim].permute(0, 2, 1).reshape(B, self.qk_dim, H, W, D)
#         # v = qkv[:, :, 2*self.qk_dim:].permute(0, 2, 1).reshape(B, self.v_dim, H, W, D)
#         q = self.q(x)
#         k = self.k(x)
#         v = self.v(x)

#         # check q, k, v data type
#         # check layer self.q weight data type
#         # print("self.q weight dtype", self.q.weight.dtype)

        


#         if self.num_heads > 1:
#             q = q.reshape(B, self.num_heads, self.qk_dim // self.num_heads, H, W, D) 
#             k = k.reshape(B, self.num_heads, self.qk_dim // self.num_heads, H, W, D)
#             v = v.reshape(B, self.num_heads, self.v_dim // self.num_heads, H, W, D)

#             q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
#             k = k.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
#             v = v.reshape(B*self.num_heads, self.v_dim // self.num_heads, H, W, D)

#         # nonliearality by seperatable kernel function elu
#         # https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
#         # q = torch.nn.functional.elu(q) + 1
#         # k = torch.nn.functional.elu(k) + 1
#         if self.qk_nonlin == 'softmax':
#             q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
#             q = q.softmax(dim=-1)
#             q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
#             k = k.softmax(dim=1)


#         # largest window size
#         # overpad to make meaningful padding for integral image for later dialated convolution
#         intergral_kv_map = build_3D_kv_integral_image(k, v, pad=self.max_window) # B, F, F, H, W, D
#         intergral_k_map = build_3D_k_integral_image(k, pad=self.max_window) # B, F, H, W, D

#         print("intergral_kv_map", intergral_kv_map.shape, intergral_kv_map.min(), intergral_kv_map.max(), intergral_kv_map.dtype)
#         print("intergral_k_map", intergral_k_map.shape, intergral_k_map.min(), intergral_k_map.max(), intergral_k_map.dtype)

        
        
#         # print("intergral_kv_map", intergral_kv_map.shape) # ([2, 32, 32, 16, 168, 168])
#         # print("intergral_k_map", intergral_k_map.shape) # ([2, 32, 8, 112, 112])
#         B_kv, F_kvk, Fkvv, H_kv, W_kv, D_kv = intergral_kv_map.shape
#         B_k, F_kk, H_k, W_k, D_k = intergral_k_map.shape
#         intergral_kv_map_reshape = intergral_kv_map.reshape(B_kv*F_kvk*Fkvv, 1, H_kv, W_kv, D_kv)
#         intergral_k_map_reshape = intergral_k_map.reshape(B_k*F_kk, 1, H_k, W_k, D_k)
        
#         # torch.zeros(B*self.num_heads, self.v_dim // self.num_heads, H,W,D, device=x.device)

#         for window_size in self.Inception_window_size:

#             assert len(window_size) == 3
#             kernel, dialation_size = build_convolve_kernel_3D_from_window_size(window_size, device=x.device)
#             kernel = kernel.unsqueeze(0).unsqueeze(0)
#             # make kernel not trainable
#             kernel.requires_grad = False

#             intergral_kv_map_padded = pad_intergral_kv_map(intergral_kv_map_reshape, self.max_window, window_size)
#             intergral_k_map_padded = pad_intergral_kv_map(intergral_k_map_reshape, self.max_window, window_size)
#             t = time.time()
#             # print("intergral_kv_map_padded", intergral_kv_map_padded.shape, intergral_kv_map_padded.min(), intergral_kv_map_padded.max())
#             # print("intergral_k_map_padded", intergral_k_map_padded.shape, intergral_k_map_padded.min(), intergral_k_map_padded.max())

#             # print("start conv3d", intergral_kv_map_padded.shape, kernel.shape, dialation_size)
#             # map to torch.float32
#             intergral_kv_map_padded = intergral_kv_map_padded.to(dtype=torch.float32)
#             intergral_k_map_padded = intergral_k_map_padded.to(dtype=torch.float32)
#             kernel = kernel.to(dtype=torch.float32)
#             with torch.cuda.amp.autocast(enabled=False):
#                 window_kv = torch.nn.functional.conv3d(input=intergral_kv_map_padded, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1)
#                 window_k = torch.nn.functional.conv3d(input=intergral_k_map_padded, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1)

            
#             # print("window_kv", window_kv.shape, window_kv.min(), window_kv.max(), window_kv.dtype)
#             # print("window_k", window_k.shape, window_k.min(), window_k.max(), window_k.dtype)

#             # print("q_reshaped", q.shape)

#             window_kv = window_kv.reshape(B_kv, F_kvk, Fkvv, -1)
#             window_k = window_k.reshape(B_k, F_kk, -1)

#             q_reshaped = q.view(B*self.num_heads, self.qk_dim // self.num_heads, -1) 
#             # print("conv3d time", time.time()-t)
            

#             Z = 1 / (torch.einsum('nld,nld->nd', q_reshaped, window_k) + 1e-6)
#             V = torch.einsum( 'nld,nlfd->nfd', q_reshaped, window_kv)
#             V = torch.einsum('nfd,nd->nfd',V, Z)
#             # start linear attnetion and normalization
#             print("V", V.shape, V.min(), V.max(), V.dtype)

#             output_attn += V.reshape(B*self.num_heads, self.v_dim // self.num_heads, H, W, D)


#         # print('output_attn', output_attn.shape, output_attn.min(), output_attn.max())
#         # asseert no NaN in output_attn
#         assert torch.all(torch.isfinite(output_attn)), "output_attn should not have inf and NaN"

#         # raise ValueError("stop here")
#         # intergral_map = build_3D_integral_image(k, v, pad=False) # B, F, F, H, W, D
#         # B1, F0, F1, H, W, D = intergral_map.shape
#         # intergral_map_reshaped = intergral_map.reshape(B1*F0*F1,1, H, W, D)

#         # with one window size -> we can get one kernel, let's say 2x2x2 kernel means window size 2x2x2
#         # after convolutoin, we get new kv_feature_map = B, F, F, D, H, W
#         # here in kv_feature_map, each FxF in this map location is the local window k v value, instead of uniform window partition
#         # then if we use same Q to do this K,V, the value BxFxDxHxW will be query vector in each dimension 
#         # if we do elementwise multiplication, it means this q with times one local window of k and v, 
#         # the problem now is standard attention, q within normalized window will time the window k
#         # but this elementwise operation of q with kv map, it means q in global normalization will time local k v window
#         # output_attn = torch.zeros(B * self.num_heads, self.v_dim // self.num_heads, H, W, D, device=x.device)

#         # for window_size in self.Inception_window_size:
#         #     assert len(window_size) == 3
#         #     kernel = build_convolve_kernel_3D_from_window_size(window_size, device=x.device)

#         #     # compute sride and padding 
            
#         #     # get same shape after conv3d
#         #     # intergral_map torch.Size([2, 1024, 8, 112, 112])
#         #     # kernel torch.Size([8, 56, 56])
#         #     kernel = kernel.unsqueeze(0).unsqueeze(0)
#         #     # print("start conv3d", intergral_map_reshaped.shape, kernel.shape)
#         #     # TODO stride for overlap ratio design
#         #     # we first try no overlap
#         #     window_kv = torch.nn.functional.conv3d(intergral_map_reshaped, kernel, stride=window_size, padding = 0)
#         #     window_kv = window_kv.reshape(B1, F0, F1, -1)
#         #     # print("window_kv", window_kv.shape) # B, F0, F1, D', H', W'
            
#         #     q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)

#         #     if self.qk_nonlin == 'softmax':
#         #         q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
#         #         q = q.softmax(dim=-1)
#         #         q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)

#         #     # start attention : q: B, F0, D, H, W,  kv: B, F0, F1, -1
#         #     for window_kv_num in range(window_kv.shape[-1]):
#         #         window_kv_feat = window_kv[..., window_kv_num] # B, F0, F1
#         #         attn = torch.einsum('bchwd,bcf->bfhwd', q, window_kv_feat)
#         #         output_attn += attn
            

#         # B1, F0, F1, H, W, D = intergral_map.shape

#         # get global feature of all tokens
        
#         # go through each window size
#         # output_attn = torch.zeros(B* self.num_heads, self.v_dim // self.num_heads, H, W, D, device=x.device)
#         # for window_size in self.Inception_window_size:
#         #     assert len(window_size) == 3
#         #     # split the H, W, D according to window size, get the top left and bottom right corner of each window
#         #     regions = create_3d_grid_via_numpy(H, W, D, window_size) # Nx6, 6 for x1, y1, z1, x2, y2, z2
#         #     regions = torch.tensor(regions, device=x.device)


#         #     window_kv = retrieval_KVs_3D_integral_image(intergral_map, regions) # B x F x F x N
#         #     window_q = retrieval_Qs_window(q, regions) # B x F x N x window_h, window_w, window_d


#         #     window_q = window_q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
#         #     if self.qk_nonlin == 'softmax':
#         #         window_q = window_q.softmax(dim=-1)
#         #     window_q = window_q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, len(regions), window_size[0], window_size[1], window_size[2])

#         #     # start attention
#         #     for region_i in range(len(regions)):
#         #         x1, y1, z1, x2, y2, z2 = regions[region_i]
#         #         window_q_i = window_q[:, :, region_i] # b x c x h x w x d
#         #         window_kv_i = window_kv[..., region_i] 
#         #         # A:  # b x c x h x w x d
#         #         # B # b x c x f 
#         #         # # target output: b x f x h x w x d
#         #         attn = torch.einsum('bchwd,bcf->bfhwd', window_q_i, window_kv_i)

#         #         output_attn[:, :, x1:x2+1, y1:y2+1, z1:z2+1] = attn
#         if self.feedforward == 'conv':
#             output_attn = output_attn.reshape(B, self.num_heads, self.v_dim // self.num_heads, H, W, D)
#             x = output_attn.reshape(B, self.v_dim, H, W, D)
#             x = self.proj(x) + x
#             x = self.proj_drop(x)
#         elif self.feedforward == 'mlp':
#             output_attn = output_attn.reshape(B,self.v_dim, -1)
#             output_attn = output_attn.permute(0, 2, 1)
#             x = self.proj(output_attn) + output_attn
#             x = x.permute(0, 2, 1).reshape(B, self.v_dim, H, W, D)

#         return x