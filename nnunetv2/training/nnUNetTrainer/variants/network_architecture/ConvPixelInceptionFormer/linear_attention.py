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

# torch.autograd.detect_anomaly(check_nan=True)


# __all__ = ["BasicUnet", "Basicunet", "basicunet", "BasicUNet"]


class TwoConv(nn.Sequential):
    """two convolutions."""

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
        print("debug input", spatial_dims, in_chns, out_chns, act, norm, bias, dropout)

        conv_0 = Convolution(
            spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


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



class BasicUnet_Encoder2Layer(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        features: Sequence[int] = (32, 32, 64, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        # now we add attention related arguments
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 4)
        print(f"BasicUNet features: {fea}.")
        # spatial_dims: int,
        # in_chns: int,
        # out_chns: int,
        # act: str | tuple,
        # norm: str | tuple,
        # bias: bool,
        # dropout: float | tuple = 0.0,
        self.conv_0 = TwoConv(spatial_dims=spatial_dims, in_chns=in_channels, out_chns=features[0], act=act, norm=norm, bias=bias, dropout=dropout)
        self.down_1 = Down(spatial_dims=spatial_dims, in_chns=fea[0], out_chns=fea[1], act=act, norm=norm, bias=bias, dropout=dropout)
        self.down_2 = Down(spatial_dims=spatial_dims, in_chns=fea[1], out_chns=fea[2], act=act, norm=norm, bias=bias, dropout=dropout)

      
    def forward(self, x: torch.Tensor):

        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)

        return x0, x1, x2




class BasicUnet_Decoder2Layer(nn.Module):
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

        self.upcat_2 = UpCat(
            spatial_dims=spatial_dims, in_chns=fea[2], 
            cat_chns=fea[1], out_chns=fea[1], 
            act=act, 
            norm=norm, 
            bias=bias, 
            dropout=dropout, 
            upsample=upsample)
        

        self.upcat_1 = UpCat(
            spatial_dims=spatial_dims, 
            in_chns=fea[1], cat_chns=fea[0], out_chns=fea[-1], act=act, norm=norm, bias=bias, dropout=dropout, upsample=upsample, halves=False)
        
        self.final_conv = Conv["conv", spatial_dims](fea[-1], out_channels, kernel_size=1)

    def forward(self, x0, x1, x2):

        u2 = self.upcat_2(x2, x1)
        u1 = self.upcat_1(u2, x0)
        logits = self.final_conv(u1)
        return logits
    

class UpCat(nn.Module):
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



class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
        pos_enc=0,
        qk_dim_compresion=1.0,
        qk_nonlin='softmax',
        qkv_func='linear', # conv, depthwise
        map_func= 'conv',
        attn_type='global', 
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        assert qk_nonlin in ['softmax', 'sigmoid', 'max']
        self.qk_nonlin = qk_nonlin
        self.qk_dim = int(dim*qk_dim_compresion)
        self.v_dim = dim
        assert qkv_func in ['linear', 'conv', 'depthwise']
        self.qkv_func = qkv_func
        assert map_func in [None, 'conv']
        self.map_func = map_func
        assert attn_type in ['global', 'window', 'linear_global', 'linear_window', 'linear_inception']
        # self.scale = qk_scale or head_dim**-0.5
        # TODO design 0: using conv for qkv 
        if qkv_func == 'linear':
            self.q = nn.Linear(dim, self.qk_dim , bias=qkv_bias)
            self.k = nn.Linear(dim, self.qk_dim , bias=qkv_bias)
            self.v = nn.Linear(dim, self.v_dim, bias=qkv_bias)
        elif qkv_func == 'conv':
            self.q = torch.nn.Conv3d(dim, self.qk_dim, 1, stride=1, padding=0, bias=False)
            self.k = torch.nn.Conv3d(dim, self.qk_dim, 1, stride=1, padding=0, bias=False)
            self.v = torch.nn.Conv3d(dim, self.v_dim, 1, stride=1, padding=0, bias=False)
        elif qkv_func == 'depthwise':
            self.q = DepthwiseSeparableConv(dim, self.qk_dim, 1, 3, bias=False)
            self.k = DepthwiseSeparableConv(dim, self.qk_dim, 1, 3, bias=False)
            self.v = DepthwiseSeparableConv(dim, self.v_dim, 1, 3, bias=False)
        else:
            raise NotImplementedError


        assert attn_drop == 0.0  # do not use
        # self.proj = nn.Linear(dim, dim)
        if map_func is None:
            self.proj = nn.Identity()
        elif map_func == 'conv':
            self.proj = nn.Conv3d(dim, dim, 1,stride=1, padding=0, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pos_enc = pos_enc
        self.input_size = input_size
        if pos_enc == 1:
            self.pos_embed = PositionalEncoding1D(dim)
        elif pos_enc == 2:
            self.pos_embed = PositionalEncoding3D(dim)
        else:
            # print("====================================")
            print("no pos enc", pos_enc)
        assert input_size[1] == input_size[2]

    def forward(self, x):
        # B, N, C = x.shape
        # print("====================================")
        #([2, 64, 4, 56, 56])
        B, C, H, W, D = x.shape
        if self.pos_enc == 2:
            pos_embed = self.pos_embed(x)
            x = x + pos_embed

        # TODO, q k v in 3D format?
        if self.qkv_func == 'conv' or self.qkv_func == 'depthwise':
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)

        x = x.view(B, C, -1)
        x = x.permute(0, 2, 1)
        if self.pos_enc == 1:
            pos_embed = self.pos_embed(x)
            x = x + pos_embed
        N = x.shape[1] 

        if self.qkv_func == 'conv' or self.qkv_func == 'depthwise':
            q = q.view(B, self.qk_dim, -1).permute(0, 2, 1).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            k = k.view(B, self.qk_dim, -1).permute(0, 2, 1).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = v.view(B, self.v_dim, -1).permute(0, 2, 1).reshape(B, N, self.num_heads, self.v_dim // self.num_heads).permute(0, 2, 1, 3)

        else:
            # print("====================================")
            # print("x shape", x.shape, "self.v_dim", self.v_dim, "self.qk_dim", self.qk_dim)
            # print 
            q = (
                self.q(x)
                .reshape(B, N, self.num_heads, self.qk_dim // self.num_heads)
                .permute(0, 2, 1, 3)
            ) # B, num_heads, N, C // num_heads

            k = (
                self.k(x)
                .reshape(B, N, self.num_heads, self.qk_dim // self.num_heads)
                .permute(0, 2, 1, 3)
            ) # B, num_heads, N, C // num_heads

            v = (
                self.v(x)
                .reshape(B, N, self.num_heads, self.v_dim // self.num_heads)
                .permute(0, 2, 1, 3)
            )
        # B, num_heads, N, C // num_heads 

        # print("x shape", x.shape)
        # print("self.num_heads,", self.num_heads)
        # print("q shape", q.shape) 
        # print("k shape", k.shape)
        # print("v shape", v.shape)
        # introduce non-linearity
        if self.qk_nonlin == 'softmax':
            q = q.softmax(dim=-1) # along feat dim will be normalized
            k = k.softmax(dim=-2) # along all tokens will be normalized
        elif self.qk_nonlin == 'sigmoid':
            q = torch.sigmoid(q)
            q = q / (q.sum(dim=-1, keepdim=True) + 1e-6)
            # print("q shape", q.shape, q.min(), q.max())
            k = torch.sigmoid(k)
            k = k / (k.sum(dim=-2, keepdim=True) + 1e-6)
            # print("k shape", k.shape, k.min(), k.max())
        elif self.qk_nonlin == 'max':
            # softmax(Q KT, dim=-1)
            # for token i, 
            # q_max = q.max(dim=-1, keepdim=True)[0]
            # k_max = k.max(dim=-2, keepdim=True)[0]
            raise NotImplementedError

        context = einsum('bhnd,bhne->bhde', k, v)
        attn = einsum('bhnd,bhde->bhne', q, context)
        # print("attn shape", attn.shape, attn.min(), attn.max())
        x = attn.transpose(1, 2).reshape(B, N, C)

        x = x.permute(0, 2, 1)
        x = x.view(B, C, H, W, D)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LinearAtnnBlock(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        feat_shape,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm3d,
        attn_func=LinearAttention,
        pos_enc=0, # int 0, 1, 2, 10, 20, 10,20 same as 1 and 2
        qk_dim_compresion=1.0,
        qk_nonlin='softmax',
        qkv_func='linear',
        map_func= 'conv',
        num_layers=1,
    ):
        super().__init__()
        # self.attn = attn_func(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_scale=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        #     input_size =feat_shape,
        #     pos_enc=pos_enc,
        #     qk_dim_compresion = qk_dim_compresion,
        #     qk_nonlin=qk_nonlin,
        #     qkv_func=qkv_func,
        #     map_func=map_func,
        # )
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        norm_layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] =attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            input_size =feat_shape,
            pos_enc=pos_enc,
            qk_dim_compresion = qk_dim_compresion,
            qk_nonlin=qk_nonlin,
            qkv_func=qkv_func,
            map_func=map_func,
            )

            norm_layers[f'encoder_layer_{i}_norm'] = norm_layer(dim)

        self.layers = nn.Sequential(layers)
        self.norms = nn.Sequential(norm_layers)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # self.norm2 = norm_layer(dim)


    def forward(self, x):        
        # x = x + self.drop_path(self.attn(x))
        # x = self.norm1(x)
        # no mlp here
        for layer, norm in zip(self.layers, self.norms):
            x = x + self.drop_path(layer(x))
            x = norm(x)

        return x

