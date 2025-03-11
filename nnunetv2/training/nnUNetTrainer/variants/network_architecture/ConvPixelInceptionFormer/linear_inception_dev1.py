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
from .basic_unet import TwoConv, DownOnly, Down

torch.autograd.detect_anomaly(check_nan=True)
from torch.utils.checkpoint import checkpoint
from torch import nn, Tensor

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
            dim=fea[1], 
            # num_heads=attn_dict['head_dims'][0],
            head_dim=attn_dict['head_dims'][0],
            # Inception_window_size=[[4, 16, 16]],
            inception=attn_dict['inception'][0],
            qk_dim_compresion=attn_dict['qk_dim_compresion'][0],
            qk_nonlin=attn_dict['qk_nonlin'],
            feat_shape=feat_shape_1,
            head_dim_qk=attn_dict.get('head_dim_qk', 32),
            kv_token_num_compression=attn_dict.get('kv_token_num_compression', False),
            global_attn=attn_dict['global_attn'][0],
            # hedgedog=attn_dict.get('hedgedog', False),
        )


        self.final_conv = Conv["conv", spatial_dims](fea[-1], out_channels, kernel_size=1)

    def forward(self, x0, x1, x2):

        u2 = self.upcat_2(x2, x1)

        # with torch.cuda.amp.autocast(enabled=False):
            # u2 = u2.to(dtype=torch.float32)
        u2 = self.layer2(u2)

        u1 = self.upcat_1(u2, x0)
        logits = self.final_conv(u1)
        return logits
    

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
        elif self.pos_enc == 0:
            self.pos_embed = None

        # patch embedding layer but is pixel embedding
        # 6 layer encoder 6 layer decoder for small model
        # 2 layer for input shape
        # 2 layer for 1/2 shape
        # 2 layer for 1/4 shape

        # self.patch_embed = nn.Sequential(
        #     nn.Conv3d(in_channels, fea[0], kernel_size=3, stride=1, padding=1),
        #     # nn.GroupNorm(1, fea[0], eps=1e-6),
        #     nn.InstanceNorm3d(fea[0], affine=True),
        #     nn.SiLU(inplace=True),
        #     nn.Conv3d(fea[0], fea[0], kernel_size=3, stride=1, padding=1),
        # )
        self.patch_embed = TwoConv(spatial_dims=3, in_chns=in_channels, out_chns=fea[0], 
                                   act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                                     norm=("instance", {"affine": True}), bias=True, dropout=dropout)


        # self.embed_feat_norm = nn.LayerNorm(fea[0])
        # self.input_norm = nn.InstanceNorm3d(in_channels, affine=True)
        # first 2 layer
        feat_shape_0 = [patch_size[0], patch_size[1], patch_size[2]]
        feat_shape_1 = [patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]
        feat_shape_2 = [patch_size[0] // 4, patch_size[1] // 4, patch_size[2] // 4]

        # self.layer_0 = LinearInceptionAtnnBlock(
        #     dim=fea[0], num_heads=attn_dict['head_dims'][0], 
        #     feat_shape=feat_shape_0,
        #     Inception_window_size=[[8, 32, 32]],
        #     qk_dim_compresion=attn_dict['qk_dim_compresion'][0],
        #     qk_nonlin=attn_dict['qk_nonlin'],
        # )
        # downsampling to feat_shape_1
        # self.down_0 = torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.down_0 = Down(spatial_dims=spatial_dims, in_chns=fea[0], out_chns=fea[1], act=act, norm=norm, bias=bias, dropout=dropout)

        # DownOnly(spatial_dims=spatial_dims, in_chns=fea[0], out_chns=fea[1], act=act, norm=norm, bias=bias, dropout=dropout)

        if attn_dict['stack_num'][0] > 0:
            self.layer_1 = nn.ModuleList(
                [
                LinearInceptioConvnAtnnBlock(
                        dim=fea[1], 
                        # num_heads=attn_dict['head_dims'][0], 
                        head_dim=attn_dict['head_dims'][0],
                        feat_shape=feat_shape_1,
                        # Inception_window_size=Inception_window_size[0], # 8x112x112
                        inception=attn_dict['inception'][0],
                        qk_dim_compresion=attn_dict['qk_dim_compresion'][0],
                        qk_nonlin=attn_dict['qk_nonlin'],
                        debug=attn_dict.get('debug', False),
                        head_dim_qk=attn_dict.get('head_dim_qk', 32),
                        kv_token_num_compression=attn_dict.get('kv_token_num_compression', False),
                        global_attn=attn_dict['global_attn'][0],
                        # hedgedog=attn_dict.get('hedgedog', False),
                    ) for i in range(attn_dict['stack_num'][0])

                ]
            )
        else:   
            self.layer_1 = TwoConv(spatial_dims=3, in_chns=fea[1], out_chns=fea[1], act=act, norm=norm, bias=bias, dropout=dropout)
            # nn.ModuleList([
            #     InvertedResidual(in_dim=fea[1], 
            #     kernel_size=3
            # )])

        # downsampling to feat_shape_2
        self.down_1 = Down(spatial_dims=spatial_dims, in_chns=fea[1], out_chns=fea[2], act=act, norm=norm, bias=bias, dropout=dropout)

        if attn_dict['stack_num'][1] > 0: 
            self.layer_2 = nn.ModuleList(
            [
                LinearInceptioConvnAtnnBlock(
                      dim=fea[2], 
                      # num_heads=attn_dict['head_dims'][1], 
                      head_dim=attn_dict['head_dims'][1],
                      feat_shape=feat_shape_2,
                    #   Inception_window_size=Inception_window_size[1], # 4x56x56
                    inception=attn_dict['inception'][1],
                      qk_dim_compresion=attn_dict['qk_dim_compresion'][1],
                      qk_nonlin=attn_dict['qk_nonlin'],
                      debug=attn_dict.get('debug', False),
                    head_dim_qk=attn_dict.get('head_dim_qk', 32),
                    kv_token_num_compression=attn_dict.get('kv_token_num_compression', False),
                    global_attn=attn_dict['global_attn'][1],
                    # hedgedog=attn_dict.get('hedgedog', False),
                 ) for i in range(attn_dict['stack_num'][1])
    
                ]
        )
        else:
            self.layer_2 = TwoConv(spatial_dims=3, in_chns=fea[2], out_chns=fea[2], act=act, norm=norm, bias=bias, dropout=dropout)

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
        # with torch.autocast(device_type='cuda', dtype=torch.float32):
            # x = self.input_norm(x)
        x_input = x
        # x = x.to(dtype=torch.float32)
        x = self.patch_embed(x)
        # print("x embed", x.shape)
        # x embed torch.Size([2, 32, 16, 224, 224])
        # x down 0 torch.Size([2, 32, 8, 112, 112])
        # x down 1 torch.Size([2, 64, 4, 56, 56])
        # x = self.layer_0(x)

        x0 = x
        x = self.down_0(x)

        # # print("x down 0", x.shape)

        # if torch.isnan(x).any():
        #     print("x", x.shape, x.min(), x.max(), x.dtype)
        #     raise ValueError("x after down_0 has NaN")

        if torch.isnan(x).any():
            print("x", x.shape, x.min(), x.max(), x.dtype)
            print("x_input", x_input.shape, x_input.min(), x_input.max(), x_input.dtype)
            raise ValueError("x after patch embd has NaN")


        B, C, H, W, D = x.shape
        if self.pos_enc == 1:
            x = x.view(B,C,-1).permute(0,2,1)
            pos_embed = self.pos_embed(x)
            pos_embed.requires_grad = False
            x = x + pos_embed
            x = x.permute(0,2,1).view(B,C,H,W,D)
        elif self.pos_enc == 2:
            pos_embed = self.pos_embed(x)
            pos_embed.requires_grad = False
            x = x + pos_embed
        

        for layer_i, blk in enumerate(self.layer_1):
            x = blk(x)
        x1 = x

        if torch.isnan(x).any():
            print("x", x.shape, x.min(), x.max(), x.dtype)
            raise ValueError("x after first convformer has NaN")
        
        x = self.down_1(x)

        if torch.isnan(x).any():
            print("x", x.shape, x.min(), x.max(), x.dtype)
            raise ValueError("x after second down has NaN")
        

 

        for layer_i, blk in enumerate(self.layer_2):
            x = blk(x)
        x2 = x

        return x0, x1, x2


class LinearInceptionConvAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=64,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=1.0,
        # input_size=(4, 14, 14),
        use_lora=0,
        qk_nonlin='softmax',
        inception=0,
        head_dim_qk=32, # when feature size is large, we can compress the qk feature dim
        kv_token_num_compression: bool = False,
        # global_attn=False, # using linear attention
    ):
        super().__init__()
        assert inception in [0,1,2,3,4], "inception should be 0, 1, 2, 3, 4"
        if dim < 64:
            head_dim = 32  # for cases when the feature size is small

        assert dim % head_dim == 0, f"dim {dim} should be divisible by num_heads {head_dim}"
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qk_nonlin = qk_nonlin
        self.inception = inception
        print("Using Inception", inception)
        # self.Inception_window_size = Inception_window_size

        qk_dim = head_dim_qk * self.num_heads # try use 32 first
        v_dim = dim

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.kv_token_num_compression = kv_token_num_compression

        # assert kv_token_num_compression == False, "not implemented yet"

        self.q = nn.Conv3d(dim, qk_dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.k = nn.Conv3d(dim, qk_dim, kernel_size=1, stride=1 if not kv_token_num_compression else 2, padding=0, bias=qkv_bias)
        # we change v to same as mednext
        # self.v = nn.Conv3d(dim, v_dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.v = nn.Conv3d(
            in_channels = dim,
            out_channels = v_dim,
            kernel_size = 3,
            stride = 1 if not kv_token_num_compression else 2,
            padding = 3//2,
            groups = dim,
            bias = True,
        )

        # we can use kernel 3 here
        # self.proj = nn.Conv3d(v_dim, v_dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.input_size = input_size
        # assert input_size[1] == input_size[2]
        # self.max_window = [max([i[0] for i in Inception_window_size]),
        #                     max([i[1] for i in Inception_window_size]), 
        #                     max([i[2] for i in Inception_window_size])]
        
        # self.global_attn = global_attn

        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                # print max and min of the weight
                print("conv3d weight", m.weight.min(), m.weight.max(), m.weight.dtype, m.weight.shape)
            
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                # print max and min of the weight
                # print("linear layer weight", m.weight.min(), m.weight.max(), m.weight.dtype, m.weight.shape)

        # raise NotImplementedError


    def forward(self, x, debug=False):
        # input x: B, C, H, W, D
        B, C, H, W, D = x.shape
        t_start = time.time()

        if self.inception == 0:
            window_h , window_w, window_d = H, W, D
            self.Inception_window_size = [[window_h, window_w, window_d]]
        elif self.inception == 1: # large window, we use half the feature as shape
            # large window: 
            window_h, window_w, window_d = max(H // 2, 1), max(W // 2, 1), max(D // 2, 1)
            self.Inception_window_size = [[window_h, window_w, window_d]]
        elif self.inception == 2: # middle window, we use half the feature as shape
            window_h, window_w, window_d = max(H // 4, 1), max(W // 4, 1), max(D // 4, 1)
            self.Inception_window_size = [[window_h, window_w, window_d]]
            # middle window:
        elif self.inception == 3: # small window, we use half the feature as shape
            # small window:
            window_h, window_w, window_d = max(H // 8, 1), max(W // 8, 1), max(D // 8, 1)
            self.Inception_window_size = [[window_h, window_w, window_d]]
        elif self.inception == 4: # no inception
            window_h_large, window_w_large, window_d_large =max(H // 2, 1), max(W // 2, 1), max(D // 2, 1)
            window_h_middle, window_w_middle, window_d_middle = max(H // 4, 1), max(W // 4, 1), max(D // 4, 1)
            window_h_small, window_w_small, window_d_small = max(H // 8, 1), max(W // 8, 1), max(D // 8, 1)
            self.Inception_window_size = [[window_h_large, window_w_large, window_d_large], [window_h_middle, window_w_middle, window_d_middle], [window_h_small, window_w_small, window_d_small]]
            # inception mix
            # window_h_large, window_w_large, window_d_large = max(H // 2, 7), max(W // 2, 7), max(D // 2, 7)
            # window_h_middle, window_w_middle, window_d_middle = max(H // 4, 5), max(W // 4, 5), max(D // 4, 5)
            # window_h_small, window_w_small, window_d_small = max(H // 8, 3), max(W // 8, 3), max(D // 8, 3)
            # self.Inception_window_size = [[window_h_large, window_w_large, window_d_large], [window_h_middle, window_w_middle, window_d_middle], [window_h_small, window_w_small, window_d_small]]
        # elif self.inception == 4: # mixed window, we use half the feature as shape
        #     self.Inception_window_size = [[H // 2, W // 2, D // 2], [H // 4, W // 4, D // 4], [H // 8, W // 8, D // 8]]
        else:
            print("inception", self.inception, "not implemented yet")
            raise NotImplementedError

        assert C == self.v_dim
        # building q, k, v
        x_3d = x 

        # debug NaN
        # if torch.isnan(x_3d).any():
        #     print("x_3d", x_3d.shape, x_3d.min(), x_3d.max(), x_3d.dtype)
        #     raise ValueError("x_3d has NaN")

        q_3d = self.q(x)
        k_3d = self.k(x)
        v_3d = self.v(x)



        _,C_,H_,W_,D_ = k_3d.shape # 

        if not self.kv_token_num_compression:  
            assert H_ == H and W_ == W and D_ == D, "k and v should have the same spatial shape"
        # if torch.isnan(q_3d).any():
        #     print("q_3d", q_3d.shape, q_3d.min(), q_3d.max(), q_3d.dtype)
        #     raise ValueError("q_3d has NaN")
        # if torch.isnan(k_3d).any():
        #     print("k_3d", k_3d.shape, k_3d.min(), k_3d.max(), k_3d.dtype)
        #     raise ValueError("k_3d has NaN")
        # if torch.isnan(v_3d).any():
        #     print("v_3d", v_3d.shape, v_3d.min(), v_3d.max(), v_3d.dtype)
        #     raise ValueError("v_3d has NaN")
        

        # linear attention idea here
        if self.qk_nonlin == 'elu':
            q_3d = torch.nn.functional.elu(q_3d) + 1 + 1e-6
            k_3d = torch.nn.functional.elu(k_3d) + 1 + 1e-6

        elif self.qk_nonlin == 'relu':
            q_3d = torch.nn.functional.relu(q_3d) + 1e-6
            k_3d = torch.nn.functional.relu(k_3d) + 1e-6

        elif self.qk_nonlin == 'sigmoid':
            q_3d = torch.nn.functional.sigmoid(q_3d) + 1e-6
            k_3d = torch.nn.functional.sigmoid(k_3d) + 1e-6

        elif self.qk_nonlin == 'softmax':
            q_3d = q_3d.reshape(B, self.qk_dim, -1)
            q_3d = q_3d.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
            q_3d = q_3d.softmax(dim=-2)  # softmax along the sequence length

            k_3d = k_3d.reshape(B, self.qk_dim, -1)
            k_3d = k_3d.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
            k_3d = k_3d.softmax(dim=-1) # softmax along the sequence length

            # print("softmax k_3d", k_3d.shape, k_3d.min(), k_3d.max(), k_3d.dtype)


            # q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
            # q = q.softmax(dim=-1)
            # q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
            # k = k.softmax(dim=1)
        else:
            raise NotImplementedError


        q_3d = q_3d.view(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
        k_3d = k_3d.view(B*self.num_heads, self.qk_dim // self.num_heads, H_, W_, D_)
        v_3d = v_3d.view(B*self.num_heads, self.v_dim // self.num_heads, H_, W_, D_)
        
  
        def create_integral_maps(k_3d, v_3d):
            with torch.autocast(device_type='cuda', dtype=torch.float32): 
                intergral_kv_map = build_3D_kv_integral_image(k_3d, v_3d)
                intergral_k_map = build_3D_k_integral_image(k_3d)
            return intergral_kv_map, intergral_k_map
        
        t_feat = time.time()
        intergral_kv_map, intergral_k_map = create_integral_maps(k_3d,v_3d)  #checkpoint(create_integral_maps, k_3d, v_3d)


        t_integral = time.time()
        # print("intergral_kv_map", intergral_kv_map.min(), intergral_kv_map.max(), intergral_kv_map.dtype)
        # print("intergral_kv_map", intergral_kv_map.shape, intergral_kv_map.min(), intergral_kv_map.max(), intergral_kv_map.dtype)
        # print("intergral_k_map", intergral_k_map.shape, intergral_k_map.min(), intergral_k_map.max(), intergral_k_map.dtype)


        intergral_kv_map = intergral_kv_map.reshape(B*self.num_heads, self.qk_dim // self.num_heads, self.v_dim // self.num_heads, H_+1, W_+1, D_+1) # pad new shape need to get new shape
        intergral_kv_map = intergral_kv_map.reshape(B*self.num_heads*(self.qk_dim // self.num_heads) * ( self.v_dim // self.num_heads), 1, H_+1, W_+1, D_+1) # B, F, F, -1


        intergral_k_map = intergral_k_map.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H_+1, W_+1, D_+1)
        intergral_k_map = intergral_k_map.reshape(B*self.num_heads*(self.qk_dim // self.num_heads), 1, H_+1, W_+1, D_+1) # B, F, F, -1

        output_attn = torch.zeros_like(x_3d).to(dtype=x.dtype).to(x.device) # B, C, H, W, D

        for window_size in self.Inception_window_size:
            assert len(window_size) == 3
            kernel, dialation_size = build_convolve_kernel_3D_from_window_size(window_size, device=x.device)
            kernel = kernel.unsqueeze(0).unsqueeze(0).to(dtype=intergral_kv_map.dtype).to(x.device)
            # make kernel not trainable
            kernel.requires_grad = False

            window_kv = torch.nn.functional.conv3d(input=intergral_kv_map, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1)
            window_k = torch.nn.functional.conv3d(input=intergral_k_map, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1) 

            window_k = window_k.clip(1e-6)


            _, _, H_i, W_i, D_i = window_kv.shape
            if H_i * W_i * D_i >= 0.1 * H * W * D: # using a small window, result large kv map, using pad
                if not self.kv_token_num_compression:        
                    target_shape = torch.tensor([H,W,D])
                else:
                    target_shape = torch.tensor([H_,W_,D_])
                window_kv_shape = torch.tensor(window_kv.shape[-3:])
                pad_start = (target_shape - window_kv_shape) // 2
                pad_end = target_shape - window_kv_shape - pad_start
                padding_size = (pad_start[2].item(), pad_end[2].item(), pad_start[1].item(), pad_end[1].item(), pad_start[0].item(), pad_end[0].item())
                pad_operation = nn.ReplicationPad3d(padding=padding_size)

                window_kv = pad_operation(window_kv)
                window_k = pad_operation(window_k)


                if self.kv_token_num_compression: 
                    assert H % H_ == 0 and W % W_ == 0 and D % D_ == 0, "H, W, D should be divisible by H_, W_, D_"
                    scale_factor = (H // H_, W // W_, D // D_)
                    window_kv = window_kv.reshape(B*self.num_heads, (self.qk_dim // self.num_heads) * (self.v_dim // self.num_heads), H_, W_, D_) # B, F, F_, -1
                    # print("window_kv", window_kv.shape)
                    window_kv = window_kv.repeat_interleave(scale_factor[0], dim=2).repeat_interleave(scale_factor[1], dim=3).repeat_interleave(scale_factor[2], dim=4)
                    # window_kv = F.interpolate(window_kv, size=(H,W,D), mode='nearest')
                    window_k = window_k.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H_, W_, D_) # B, F, -1
                    # window_k = F.interpolate(window_k, size=(H,W,D), mode='nearest')
                    window_k = window_k.repeat_interleave(scale_factor[0], dim=2).repeat_interleave(scale_factor[1], dim=3).repeat_interleave(scale_factor[2], dim=4)

                    # print("window_kv", window_kv.shape, window_kv.min(), window_kv.max(), window_kv.dtype)
                    # print("window_k", window_k.shape, window_k.min(), window_k.max(), window_k.dtype)

                window_kv = window_kv.view(B*self.num_heads, self.qk_dim // self.num_heads, self.v_dim // self.num_heads, H, W, D) # B, F, F_, -1
                window_k = window_k.view(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D) # B, F, -1
                # q_3d: B, F, H, W, D
                # print("q_3d", q_3d.shape, q_3d.min(), q_3d.max(), q_3d.dtype)
                # print("window_kv", window_kv.shape, window_kv.min(), window_kv.max(), window_kv.dtype)
                attn = torch.einsum('bchwd,bcfhwd->bfhwd', q_3d, window_kv)
                norm = torch.einsum('bchwd,bchwd->bhwd', q_3d, window_k)

                # print("attn", attn.dtype)
                # print("norm", norm.dtype)

                attn = attn / (norm.unsqueeze(1) + 1e-6)  # B, F, H, W, D

                # print("norm attn", attn.dtype)
                attn = attn.reshape(B,self.v_dim, H, W, D)

                if torch.isnan(attn).any() or torch.isinf(attn).any():
                    print("attn", attn.shape, attn.min(), attn.max(), attn.dtype)
                    print("norm", norm.shape, norm.min(), norm.max(), norm.dtype)
                    print("window_kv", window_kv.shape, window_kv.min(), window_kv.max(), window_kv.dtype)
                    print("window_k", window_k.shape, window_k.min(), window_k.max(), window_k.dtype)
                    raise ValueError(f"attn {window_size} has NaN/inf")
                    # attn torch.Size([2, 32, 8, 112, 112]) tensor(-inf, device='cuda:0', dtype=torch.float16, grad_fn=<MinBackward1>) tensor(inf, device='cuda:0', dtype=torch.float16, grad_fn=<MaxBackward1>) torch.float16

                output_attn = output_attn + attn
            else: # using a large window, result small kv map, using for loop
                #no pad implementation
                _, _, H_i, W_i, D_i = window_kv.shape

                window_kv = window_kv.view(B*self.num_heads, self.qk_dim // self.num_heads, self.v_dim // self.num_heads, H_i, W_i, D_i) # B, F, F, -1
                window_k_pad = window_k.view(B*self.num_heads, self.qk_dim // self.num_heads, H_i, W_i, D_i) # B, F, -1

                q_3d_reshape = q_3d.permute(0,2,3,4,1).reshape(B*self.num_heads, H*W*D, self.qk_dim // self.num_heads)
                window_kv_reshape = window_kv.reshape(B*self.num_heads, self.qk_dim // self.num_heads, self.v_dim // self.num_heads , H_i*W_i*D_i) #
                window_k_reshape = window_k_pad.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H_i*W_i*D_i)

                
                new_length = H_i*W_i*D_i
                # for loop
                for count in range(new_length):
                    window_kv_reshape_j = window_kv_reshape[...,count] # B,F,F_
                    window_k_reshape_j = window_k_reshape[...,count] # B,F
                    attn = torch.einsum('bnd,bdj->bnj', q_3d_reshape, window_kv_reshape_j)# B, H*W*D, F_
                    norm = torch.einsum('bnd,bd->bn', q_3d_reshape, window_k_reshape_j)  # B, H*W*D
                    # print("attn_all", attn_all.dtype)
                    # print("norm_all", norm_all.dtype)
                    attn = attn / (norm.unsqueeze(-1) + 1e-6)  # B, H*W*D, F_
                    # print("noramlize attn_all", attn_all.dtype)
                    attn = attn.reshape(B, self.num_heads, H, W, D, self.v_dim // self.num_heads).permute(0, 1, 5, 2, 3, 4).reshape(B, self.v_dim, H, W, D)
                    output_attn = output_attn + attn

                    if torch.isnan(attn).any() or torch.isinf(attn).any():
                        print("attn", attn.shape, attn.min(), attn.max(), attn.dtype)
                        print("norm", norm.shape, norm.min(), norm.max(), norm.dtype)
                        print("window_kv", window_kv.shape, window_kv.min(), window_kv.max(), window_kv.dtype)
                        print("window_k", window_k.shape, window_k.min(), window_k.max(), window_k.dtype)
                        raise ValueError(f"attn {window_size} in loop has NaN/inf")
        

                # non for loop
                # similar speed as for loop
                # attn_all = torch.einsum('bnd,bdfm->bnfm', q_3d_reshape, window_kv_reshape)# B, H*W*D, F_, H_i*W_i*D_i
                # norm_all = torch.einsum('bnd,bdm->bnm', q_3d_reshape, window_k_reshape)  # B, H*W*D, H_i*W_i*D_i
                # attn_all = attn_all / (norm_all.unsqueeze(-2) + 1e-6)  # B, H*W*D, F_, H_i*W_i*D_i
                # attn_all = attn_all.sum(dim=-1) # B, H*W*D, F_
                # output_attn = output_attn + attn_all.reshape(B, self.num_heads, H, W, D, self.v_dim // self.num_heads).permute(0, 1, 5, 2, 3, 4).reshape(B, self.v_dim, H, W, D)

            t_inception = time.time()
            assert len(self.Inception_window_size) > 0
            output_attn = output_attn / len(self.Inception_window_size)
            # output_attn = self.proj(output_attn)
            # output_attn = self.proj_drop(output_attn)

        # if torch.isnan(output_attn).any() or torch.isinf(output_attn).any():
        #     print("output_attn", output_attn.shape, output_attn.min(), output_attn.max(), output_attn.dtype)
        #     print("window_kv", window_kv.shape, window_kv.min(), window_kv.max(), window_kv.dtype)
        #     print("window_k", window_k.shape, window_k.min(), window_k.max(), window_k.dtype)
        #     print("attn", attn.shape, attn.min(), attn.max(), attn.dtype)
        #     raise ValueError("output_attn has NaN / inf")
        
        # after iteration, release the integral map to save memory
        del intergral_kv_map
        del intergral_k_map
        torch.cuda.empty_cache()

        if debug:
            print("t_feat", t_feat - t_start)
            print("t_integral", t_integral - t_feat)
            print("t_inception", t_inception - t_integral)
            # inception 0 
            # t_feat 0.00043582916259765625
            # t_integral 0.01801896095275879
            # t_inception 0.010710000991821289
            # backward time with grad_scaler: 1.1402969360351562

            # global
            # backward time with grad_scaler: 0.10748600959777832

        return output_attn


class LinearInceptioConvnAtnnBlock(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        head_dim,
        feat_shape,
        inception,
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
        head_dim_qk=32, # when feature size is large, we can compress the qk feature dim
        global_attn=False, # using linear attention
        # hedgedog=False,
        debug=False,
        kv_token_num_compression: bool = False,
    ):
        super().__init__()
        # inception=0,
        # head_dim_qk=32, # when feature size is large, we can compress the qk feature dim
        # kv_token_num_compression: bool = False,
        # global_attn=True, # using linear attention

        # self.norm1 = nn.InstanceNorm3d(dim, affine=True)
        # norm_layer(1, dim, eps=1e-6)
        
        if global_attn:
            self.attn = LinearConvAttention(
                dim=dim,
                head_dim=head_dim,
                qkv_bias=qkv_bias,
                # input_size=feat_shape,
                qk_nonlin=qk_nonlin,
                head_dim_qk=head_dim_qk,
            )
        else:
            self.attn = LinearInceptionConvAttention(
                dim=dim,
                head_dim=head_dim,
                qkv_bias=qkv_bias,
                # input_size=feat_shape,
                use_lora=0,
                qk_nonlin=qk_nonlin,
                inception=inception,
                head_dim_qk=head_dim_qk,
                kv_token_num_compression=kv_token_num_compression,
            )
        # norm_layer(1,dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # feedforward set to be InvertedResidual
        
        self.conv2 = nn.Conv3d(
            in_channels = dim,
            out_channels = 4*dim,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        self.act = nn.LeakyReLU( negative_slope=0.1, inplace=True)
        self.conv3 = nn.Conv3d(
            in_channels = 4*dim,
            out_channels = dim,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        # initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):        
        # input x: B, C, H, W, D
        B, C, H, W, D = x.shape
        # norm 
        # if torch.isnan(x).any():
        #     print("x", x.shape, x.min(), x.max(), x.dtype)
        #     raise ValueError("x has NaN")
        
        attn = self.drop_path(self.attn(x))

        # if torch.isnan(attn).any():
        #     print("attn layer", x.shape, x.min(), x.max(), x.dtype)
        #     raise ValueError("attn layer has NaN")

        # print("second mlp", attn.shape, attn.min(), attn.max(), attn.dtype)
        attn = self.act(self.conv2(attn))

        # if torch.isnan(attn).any():
        #     print("second mlp", x.shape, x.min(), x.max(), x.dtype)
        #     raise ValueError("second mlp has NaN")
        
        attn = self.conv3(attn)

        # if torch.isnan(attn).any():
        #     print("third mlp", x.shape, x.min(), x.max(), x.dtype)
        #     raise ValueError("third mlp has NaN")
        x = x + attn
        # x = x + self.drop_path(self.irb(x))
        return x




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



def build_3D_k_integral_image(k):
    # k: B, F, H, W, D
    B, F, D, H, W = k.shape
    # k = k.to(dtype=torch.float16)
    output = torch.cumsum(k, dim=2).to(k.dtype)
    output = torch.cumsum(output, dim=3).to(k.dtype)
    output = torch.cumsum(output, dim=4).to(k.dtype)

    output = torch.nn.functional.pad(output, (1,0,1,0,1,0), mode='constant', value=0)
    assert torch.isnan(output).any() == False, "output has NaN"
    assert torch.isinf(output).any() == False, "output has inf"
    return output

def build_3D_kv_integral_image(k, v):
    # k: B, F, H, W, D
    # v: B, F_, H, W, D
    # output: B, F, F, D+1, H+1, W+1
    # save memory
    # k = k.to(dtype=torch.float16)
    # v = v.to(dtype=torch.float16)

    assert k.shape[-3:] == v.shape[-3:], "k and v should have the same spatial shape"
    B, F, D, H, W = k.shape
    B, F_, _, _, _ = v.shape

    # k_ = k.unsqueeze(2) # B, F, 1, H, W, D
    # v_ = v.unsqueeze(1) # B, 1, F_, H, W, D
    # kv_product = torch.matmul(k_, v_) # B, F, F_, H, W, D
    kv_product = torch.einsum('bfhwd,bghwd->bfghwd', k, v).to(k.dtype)
    # .to(k.dtype) enable float16, otherwise will autocast to float32

    # is already floaat16
    kv_product = torch.cumsum(kv_product, dim=3).to(k.dtype)
    # print("kv_product0", kv_product.shape, kv_product.min(), kv_product.max(), kv_product.dtype)
    kv_product = torch.cumsum(kv_product, dim=4).to(k.dtype)
    # print("kv_product1", kv_product.shape, kv_product.min(), kv_product.max(), kv_product.dtype)
    kv_product = torch.cumsum(kv_product, dim=5).to(k.dtype)
    # print("kv_product2", kv_product.shape, kv_product.min(), kv_product.max(), kv_product.dtype)
    kv_product = torch.nn.functional.pad(kv_product, (1,0,1,0,1,0), mode='constant', value=0)

    assert torch.isnan(kv_product).any() == False, "kv_product has NaN"
    assert torch.isinf(kv_product).any() == False, "kv_product has inf"
    return kv_product

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
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)
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
    

class LinearConvAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=64,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        # input_size=(4, 14, 14),
        qk_nonlin='softmax',
        
        head_dim_qk=32, # when feature size is large, we can compress the qk feature dim
        global_attn=True, # using linear attention
    ):
        super().__init__()
        if dim < 64:
            head_dim = 32

        assert dim % head_dim == 0, f"dim {dim} should be divisible by num_heads {head_dim}"
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qk_nonlin = qk_nonlin
        # self.Inception_window_size = Inception_window_size

        qk_dim = head_dim_qk * self.num_heads
        v_dim = dim

        self.qk_dim = qk_dim
        self.v_dim = v_dim

        self.q = nn.Conv3d(dim, qk_dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.k = nn.Conv3d(dim, qk_dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        # self.v = nn.Conv3d(dim, v_dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.v = nn.Conv3d(
            in_channels = dim,
            out_channels = v_dim,
            kernel_size = 3,
            stride = 1,
            padding = 3//2,
            groups = dim,
        )
        # we can use kernel 3 here
        self.proj = nn.Conv3d(v_dim, v_dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.input_size = input_size
        # assert input_size[1] == input_size[2]
        # self.max_window = [max([i[0] for i in Inception_window_size]),
        #                     max([i[1] for i in Inception_window_size]), 
        #                     max([i[2] for i in Inception_window_size])]
        
        self.global_attn = global_attn

        # initialize the weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
            
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # input x: B, C, H, W, D
        B, C, H, W, D = x.shape
        assert C == self.v_dim, f"input feature dim {C} should be equal to v_dim {self.v_dim}"
        # building q, k, v
        x_3d = x 

        # check non-mixed precision
        # print("x_3d", x_3d.dtype)
        # debug NaN
        # if torch.isnan(x_3d).any():
        #     print("x_3d", x_3d.shape, x_3d.min(), x_3d.max(), x_3d.dtype)
        #     raise ValueError("x_3d has NaN")

        q_3d = self.q(x)
        k_3d = self.k(x)
        v_3d = self.v(x)

        # print("v_3d", v_3d.dtype)

        # if torch.isnan(q_3d).any():
        #     print("q_3d", q_3d.shape, q_3d.min(), q_3d.max(), q_3d.dtype)
        #     raise ValueError("q_3d has NaN")
        # if torch.isnan(k_3d).any():
        #     print("k_3d", k_3d.shape, k_3d.min(), k_3d.max(), k_3d.dtype)
        #     raise ValueError("k_3d has NaN")
        # if torch.isnan(v_3d).any():
        #     print("v_3d", v_3d.shape, v_3d.min(), v_3d.max(), v_3d.dtype)
        #     raise ValueError("v_3d has NaN")
        

        # linear attention idea here
        if self.qk_nonlin == 'elu':
            q_3d = torch.nn.functional.elu(q_3d) + 1 + 1e-6
            k_3d = torch.nn.functional.elu(k_3d) + 1 + 1e-6

        elif self.qk_nonlin == 'softmax':
            q_3d = q_3d.reshape(B, self.qk_dim, -1)
            q_3d = q_3d.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
            q_3d = q_3d.softmax(dim=-2) 

            k_3d = k_3d.reshape(B, self.qk_dim, -1)
            k_3d = k_3d.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
            k_3d = k_3d.softmax(dim=-1)# softmax along the sequence length

            # print("softmax k_3d", k_3d.shape, k_3d.min(), k_3d.max(), k_3d.dtype)


            # q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, -1)
            # q = q.softmax(dim=-1)
            # q = q.reshape(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
            # k = k.softmax(dim=1)
        else:
            raise NotImplementedError

        # print("norm q_3d", q_3d.dtype)

        q_3d = q_3d.view(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
        k_3d = k_3d.view(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
        v_3d = v_3d.view(B*self.num_heads, self.v_dim // self.num_heads, H, W, D)

        # do global linear atteention, no inception idea
        q_3d = q_3d.reshape(B, self.num_heads, self.qk_dim // self.num_heads, -1).permute(0, 1, 3, 2)
        k_3d = k_3d.reshape(B, self.num_heads, self.qk_dim // self.num_heads, -1).permute(0, 1, 3, 2)
        v_3d = v_3d.reshape(B, self.num_heads, self.v_dim // self.num_heads, -1).permute(0, 1, 3, 2)

        # print("q_3d", q_3d.shape)
        # print("k_3d", k_3d.shape)
        # print("v_3d", v_3d.shape)

        kv_3d = (k_3d.transpose(-2, -1) @ v_3d)  # B, num_heads, F, F
        k_3d = k_3d.sum(dim=-2, keepdim=True) # B, num_heads, 1, F

        # print("context", kv_3d.dtype)
        # print("k_3d", k_3d.dtype)


        # print("sum k_3d", k_3d.shape, k_3d.min(), k_3d.max(), k_3d.dtype)
        # print("kv_3d", kv_3d.shape, kv_3d.min(), kv_3d.max(), kv_3d.dtype)
        attn = q_3d @ kv_3d  #* self.scale # B, num_heads, H*W*D, F
        # print("attn", attn.shape, attn.min(), attn.max(), attn.dtype)
        # print("q_3d", q_3d.shape, q_3d.min(), q_3d.max(), q_3d.dtype)
        # print("k_3d", k_3d.shape, k_3d.min(), k_3d.max(), k_3d.dtype)

        # print("attn results", attn.shape, attn.min(), attn.max(), attn.dtype)
        # print("attn", attn.dtype)

        Z = q_3d @ k_3d.transpose(-2, -1)  # B, num_heads, H*W*D, F @ B, num_heads, F, 1 = B, num_heads, H*W*D, 1

        # print("normalization", Z.shape, Z.min(), Z.max(), Z.dtype)

        attn = attn / (Z+ 1e-6)


        # print("normalization results", attn.dtype)

        attn = attn.permute(0, 1, 3, 2).reshape(B, C, H, W, D)

        # print("output_attn", output_attn.shape, output_attn.min(), output_attn.max(), output_attn.dtype)
        # NaN check here
        if torch.isnan(attn).any():
            print("output_attn", attn.shape, attn.min(), attn.max(), attn.dtype)
            raise ValueError("output_attn has NaN")
        
        if torch.isinf(attn).any():
            print("output_attn", attn.shape, attn.min(), attn.max(), attn.dtype)
            raise ValueError("output_attn has inf")
        return attn



class HedgeDogConvAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=64,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        # input_size=(4, 14, 14),
        head_dim_qk=32, # when feature size is large, we can compress the qk feature dim
        **kwargs
    ):
        super().__init__()
        if dim < 64:
            head_dim = 32

        assert dim % head_dim == 0, f"dim {dim} should be divisible by num_heads {head_dim}"
        self.num_heads = dim // head_dim

        # self.Inception_window_size = Inception_window_size

        qk_dim = head_dim_qk * self.num_heads
        v_dim = dim

        self.qk_dim = qk_dim
        self.v_dim = v_dim

        self.q = nn.Conv3d(dim, qk_dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.k = nn.Conv3d(dim, qk_dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        # self.v = nn.Conv3d(dim, v_dim, kernel_size=1, stride=1, padding=0, bias=qkv_bias)
        self.v = nn.Conv3d(
            in_channels = dim,
            out_channels = v_dim,
            kernel_size = 3,
            stride = 1,
            padding = 3//2,
            groups = dim,
        )

        self.mlp_q = HedgeHogModule(head_dim)
        self.mlp_k = HedgeHogModule(head_dim)
        self.mlp_v = HedgeHogModule(head_dim)

    def forward(self, x):
        # input x: B, C, H, W, D
        B, C, H, W, D = x.shape
        assert C == self.v_dim, f"input feature dim {C} should be equal to v_dim {self.v_dim}"
        # building q, k, v
        x_3d = x 

        q_3d = self.q(x)
        k_3d = self.k(x)
        v_3d = self.v(x)
        


        # print("norm q_3d", q_3d.dtype)

        q_3d = q_3d.view(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
        k_3d = k_3d.view(B*self.num_heads, self.qk_dim // self.num_heads, H, W, D)
        v_3d = v_3d.view(B*self.num_heads, self.v_dim // self.num_heads, H, W, D)

        # do global linear atteention, no inception idea
        q_1d = q_3d.reshape(B, self.num_heads, self.qk_dim // self.num_heads, -1).permute(0, 1, 3, 2) # B, H, B, D
        k_1d = k_3d.reshape(B, self.num_heads, self.qk_dim // self.num_heads, -1).permute(0, 1, 3, 2)
        v_1d = v_3d.reshape(B, self.num_heads, self.v_dim // self.num_heads, -1).permute(0, 1, 3, 2)

        q_1d = self.mlp_q(q_1d)
        k_1d = self.mlp_k(k_1d)
        v_1d = self.mlp_v(v_1d)

        attn = quadratic_linear_attn(q_1d, k_1d)
        attn = attn @ v_1d


        attn = attn.permute(0, 1, 3, 2).reshape(B, C*2, H, W, D)

        # print("output_attn", output_attn.shape, output_attn.min(), output_attn.max(), output_attn.dtype)
        # NaN check here
        if torch.isnan(attn).any():
            print("output_attn", attn.shape, attn.min(), attn.max(), attn.dtype)
            raise ValueError("output_attn has NaN")
        
        if torch.isinf(attn).any():
            print("output_attn", attn.shape, attn.min(), attn.max(), attn.dtype)
            raise ValueError("output_attn has inf")
        return attn

def quadratic_linear_attn(
    q: Tensor,
    k: Tensor,
):
    qk = torch.einsum("bhmd, bhnd -> bhmn", q, k)
    return qk / (qk.sum(dim=-1, keepdim=True) + 1e-6 )


class HedgeHogModule(nn.Module):
    """
    HedgeHogModule is a PyTorch module that applies linear transformation
    followed by an activation function to the input tensor.

    Args:
        head_dim (int): The dimension of the input tensor.
        activation (str, optional): The activation function to be applied.
            Defaults to "exp".

    Attributes:
        head_dim (int): The dimension of the input tensor.
        activation (str): The activation function to be applied.
        layer (nn.Linear): The linear transformation layer.

    Methods:
        init_weights: Initializes the weights of the linear layer.
        forward: Performs forward pass through the module.

    """

    def __init__(
        self,
        dim: int,
        activation: str = "exp",
    ):
        super().__init__()
        self.dim = dim
        self.activation = activation
        self.layer = nn.Linear(dim, dim)
        self.init_weights_()

    def init_weights_(self):
        nn.init.eye_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass through the module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying linear transformation
            and activation function.

        """
        x = self.layer(x)  # Shape BATCH, HEADS, SEQLEN, DIMENSION
        return torch.cat([torch.exp(x), torch.exp(-x)], dim=1)