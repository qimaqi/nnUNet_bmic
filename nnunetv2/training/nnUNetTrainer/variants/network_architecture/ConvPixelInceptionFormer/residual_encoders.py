
import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.residual import BottleneckD, BasicBlockD
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath # ,  Mlp
from torch import nn, einsum

import loralib as lora
from functools import partial
import time


class StackedResidualBlocksInception(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 Inception_head_num: int = 1,
                 feat_shape: List[int] = None, # feat_shape after first cnn block
                 pos_enc: int = 0, # 1 and 2, 10 and 20
                 qk_dim_compresion: float = 1.0,
                 qk_nonlin: str = 'softmax',
                 qkv_func: str = 'linear',
                 map_func: str = 'conv',
                 ):
        """
        Stack multiple instances of block.

        :param n_blocks: number of residual blocks
        :param conv_op: nn.ConvNd class
        :param input_channels: only relevant for forst block in the sequence. This is the input number of features.
        After the first block, the number of features in the main path to which the residuals are added is output_channels
        :param output_channels: number of features in the main path to which the residuals are added (and also the
        number of features of the output)
        :param kernel_size: kernel size for all nxn (n!=1) convolutions. Default: 3x3
        :param initial_stride: only affects the first block. All subsequent blocks have stride 1
        :param conv_bias: usually False
        :param norm_op: nn.BatchNormNd, InstanceNormNd etc
        :param norm_op_kwargs: dictionary of kwargs. Leave empty ({}) for defaults
        :param dropout_op: nn.DropoutNd, can be None for no dropout
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param block: BasicBlockD or BottleneckD
        :param bottleneck_channels: if block is BottleneckD then we need to know the number of bottleneck features.
        Bottleneck will use first 1x1 conv to reduce input to bottleneck features, then run the nxn (see kernel_size)
        conv on that (bottleneck -> bottleneck). Finally the output will be projected back to output_channels
        (bottleneck -> output_channels) with the final 1x1 conv
        :param stochastic_depth_p: probability of applying stochastic depth in residual blocks
        :param squeeze_excitation: whether to apply squeeze and excitation or not
        :param squeeze_excitation_reduction_ratio: ratio by how much squeeze and excitation should reduce channels
        respective to number of out channels of respective block
        """
        super().__init__()
        assert n_blocks > 0, 'n_blocks must be > 0'
        assert block in [BasicBlockD, BottleneckD], 'block must be BasicBlockD or BottleneckD'
        self.pos_enc = pos_enc
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks
        if not isinstance(bottleneck_channels, (tuple, list)):
            bottleneck_channels = [bottleneck_channels] * n_blocks

        if block == BasicBlockD:
            # downsample at beginning of block
            blocks_list = []
            blocks_list.append(
                    block(conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias,
                    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                    squeeze_excitation, squeeze_excitation_reduction_ratio)
            )
            blocks_list.append(
                LinearAtnnBlock(output_channels[0], Inception_head_num, qkv_bias=False, qk_scale=None, act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm3d, attn_func=LinearAttention, feat_shape=feat_shape, pos_enc=pos_enc, qk_dim_compresion=qk_dim_compresion, qk_nonlin=qk_nonlin, qkv_func=qkv_func, map_func=map_func)   
            )
            for n in range(1, n_blocks):
                blocks_list.append(
                    block(conv_op, output_channels[n - 1], output_channels[n], kernel_size, 1, conv_bias, norm_op,
                          norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                          squeeze_excitation, squeeze_excitation_reduction_ratio)
                )
                blocks_list.append(
                    LinearAtnnBlock(output_channels[n], Inception_head_num, qkv_bias=False, qk_scale=None, act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm3d, attn_func=LinearAttention, feat_shape=feat_shape, pos_enc=pos_enc if pos_enc == 10 or pos_enc == 20 else 0, qk_dim_compresion=qk_dim_compresion, qk_nonlin=qk_nonlin, qkv_func=qkv_func, map_func=map_func)
                )

                # *[block(conv_op, output_channels[n - 1], output_channels[n], kernel_size, 1, conv_bias, norm_op,
                #         norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                #         squeeze_excitation, squeeze_excitation_reduction_ratio) for n in range(1, n_blocks)]

            blocks = nn.Sequential(*blocks_list)

        else:
            blocks = nn.Sequential(
                block(conv_op, input_channels, bottleneck_channels[0], output_channels[0], kernel_size,
                      initial_stride, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                      nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], bottleneck_channels[n], output_channels[n], kernel_size,
                        1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation,
                        squeeze_excitation_reduction_ratio) for n in range(1, n_blocks)]
            )
        self.blocks = blocks
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)
        self.output_channels = output_channels[-1]

    def forward(self, x):
        return self.blocks(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output = self.blocks[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.blocks[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


# 1d 
def create_sinusoidal_embeddings(n_pos, dim):
    B, C, H, W, D = x.shape
    t0 = time.time()
    out = torch.zeros(n_pos, dim)
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    print("create_sinusoidal_embeddings time", time.time()-t0)
    return out

# 3d

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()

        # torch.autograd.set_detect_anomaly(True)
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        inv_freq = inv_freq.to(dtype=torch.float64)
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """

        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)

        # print("emb_x", emb_x.shape, emb_x.min(), emb_x.max())
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        print("====================================")
        print("using 3d pos enc")
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, ch, x, y, z)
        :return: Positional Encoding Matrix of size (batch_size, ch, x, y, z)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, orig_ch, x, y, z = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = torch.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        self.cached_penc = self.cached_penc.permute(0, 4, 1, 2, 3)
        return self.cached_penc

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, bias=False):
        super().__init__()
        
        if isinstance(kernel_size, list):
            padding = [i//2 for i in kernel_size]
        else:
            padding = kernel_size // 2

        self.depthwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias
        )
        self.pointwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias
        )
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

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
        qkv_func='linear', # conv
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
        assert qkv_func in ['linear', 'conv','depthwise']
        self.qkv_func = qkv_func
        assert map_func in [None, 'conv']
        self.map_func = map_func
        assert attn_type in ['global', 'window', 'inception']
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
        if pos_enc == 1 or pos_enc == 10:
            self.pos_embed = PositionalEncoding1D(dim)
        elif pos_enc == 2 or pos_enc == 20:
            self.pos_embed = PositionalEncoding3D(dim)
        else:
            # print("====================================")
            print("no pos enc", pos_enc)
        assert input_size[1] == input_size[2]

    def forward(self, x):
        # B, N, C = x.shape
        B, C, H, W, D = x.shape
        if self.pos_enc == 2 or self.pos_enc == 20:
            pos_embed = self.pos_embed(x)
            x = x + pos_embed

        # TODO, q k v in 3D format?
        if self.qkv_func == 'conv' or self.qkv_func == 'depthwise':
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            # print("q shape", q.shape) # q shape torch.Size([2, 32, 64, 160, 160])
            # print("k shape", k.shape)
            # print("v shape", v.shape)


        x = x.view(B, C, -1)
        x = x.permute(0, 2, 1)
        if self.pos_enc == 1 or self.pos_enc == 10:
            pos_embed = self.pos_embed(x)
            x = x + pos_embed
        N = x.shape[1] 

        if self.qkv_func == 'conv' or self.qkv_func == 'depthwise':
            q = q.view(B, self.qk_dim, -1).permute(0, 2, 1).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads)
            k = k.view(B, self.qk_dim, -1).permute(0, 2, 1).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads)
            v = v.view(B, self.v_dim, -1).permute(0, 2, 1).reshape(B, N, self.num_heads, self.v_dim // self.num_heads)

        else:
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
            k = torch.sigmoid(k)
        elif self.qk_nonlin == 'max':
            # softmax(Q KT, dim=-1)
            # for token i, 
            # q_max = q.max(dim=-1, keepdim=True)[0]
            # k_max = k.max(dim=-2, keepdim=True)[0]
            raise NotImplementedError

        context = einsum('bhnd,bhne->bhde', k, v)
        attn = einsum('bhnd,bhde->bhne', q, context)
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
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm3d,
        attn_func=LinearAttention,
        feat_shape=None,
        pos_enc=0, # int 0, 1, 2, 10, 20, 10,20 same as 1 and 2
        qk_dim_compresion=1.0,
        qk_nonlin='softmax',
        qkv_func='linear',
        map_func= 'conv',
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
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
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # self.norm2 = norm_layer(dim)


    def forward(self, x):        
        x = x + self.drop_path(self.attn(x))
        x = self.norm1(x)
        # no mlp here
        return x


class ResidualEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 patch_size: List[int] = None,
                 pos_enc: int = 0, # 0 not posenc, 1 use 1d, 2: use 3d, 10, use 1d but on all layer, 20 use 3d but on all layer
                 qk_dim_compresion: float = 1.0,
                 qk_nonlin: str = 'softmax',
                qkv_func: str = 'linear',
                map_func: str = 'conv',
                 ):
        """

        :param input_channels:
        :param n_stages:
        :param features_per_stage: Note: If the block is BottleneckD, then this number is supposed to be the number of
        features AFTER the expansion (which is not coded implicitly in this repository)! See todo!
        :param conv_op:
        :param kernel_sizes:
        :param strides:
        :param n_blocks_per_stage:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param block:
        :param bottleneck_channels: only needed if block is BottleneckD
        :param return_skips: set this to True if used as encoder in a U-Net like network
        :param disable_default_stem: If True then no stem will be created. You need to build your own and ensure it is executed first, see todo.
        The stem in this implementation does not so stride/pooling so building your own stem is a necessity if you need this.
        :param stem_channels: if None, features_per_stage[0] will be used for the default stem. Not recommended for BottleneckD
        :param pool_type: if conv, strided conv will be used. avg = average pooling, max = max pooling
        """
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        assert len(
            bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        assert pos_enc in [0, 1, 2, 10, 20], "pos_enc must be 0, 1, 2, 10, 20"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        # build a stem, Todo maybe we need more flexibility for this in the future. For now, if you need a custom
        #  stem you can just disable the stem and build your own.
        #  THE STEM DOES NOT DO STRIDE/POOLING IN THIS IMPLEMENTATION
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            input_channels = stem_channels
        else:
            self.stem = None

        # now build the network
        stages = []
        feature_map_size = []
        previous_shape = patch_size


        
        # Linear attention information
        # 6 stages
            # num_heads of medformer [1,4,8,16, 8,4,1,1],
        # 6 stage feat num [32, 64, 128, 256, 320, 320]
        # head [1, 2, 4, 8, 10, 10 ]
        print("====================================")
        print("n_stages", n_stages)
        print("n_blocks_per_stage", n_blocks_per_stage)
        encoder_inception_head_num = [1, 2, 4, 8, 10, 10 ] # less feature dimension use less head
        # "n_blocks_per_stage": [
        #     1,
        #     3,
        #     4,
        #     6,
        #     6,
        #     6
        # ],
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1
            # need shape for positional encoding, we assume that the input is a square
            feat_shape = [i // j for i, j in zip(previous_shape, maybe_convert_scalar_to_list(conv_op, strides[s]))]
            if pos_enc ==10 or pos_enc == 20:
                use_pos_enc = True 
            elif pos_enc == 1 or pos_enc == 2:
                use_pos_enc = s == 0
            else:
                use_pos_enc = 0

            stage = StackedResidualBlocksInception(
                n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                squeeze_excitation=squeeze_excitation,
                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio,
                Inception_head_num=encoder_inception_head_num[s],
                feat_shape=feat_shape,
                pos_enc= pos_enc if use_pos_enc else 0,
                qk_dim_compresion=qk_dim_compresion,
                qk_nonlin=qk_nonlin,
                qkv_func=qkv_func,
                map_func=map_func,
            )

            if pool_op is not None:
                stage = nn.Sequential(pool_op(strides[s]), stage)

            stages.append(stage)
            input_channels = features_per_stage[s]
        
        # def cal_feat_map_size(input_size, stride):

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            # print("x shape", x.shape)
            ret.append(x)


        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


if __name__ == '__main__':
    data = torch.rand((1, 3, 128, 160))

    model = ResidualEncoder(3, 5, (2, 4, 6, 8, 10), nn.Conv2d, 3, ((1, 1), 2, (2, 2), (2, 2), (2, 2)), 2, False,
                            nn.BatchNorm2d, None, None, None, nn.ReLU, None, stem_channels=7)
    # import hiddenlayer as hl

    # g = hl.build_graph(model, data,
    #                    transforms=None)
    # g.save("network_architecture.pdf")
    # del g

    print(model.compute_conv_feature_map_size((128, 160)))


        #         "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
        #         "arch_kwargs": {
        #             "n_stages": 8,
        #             "features_per_stage": [
        #                 32,
        #                 64,
        #                 128,
        #                 256,
        #                 512,
        #                 512,
        #                 512,
        #                 512
        #             ],
        #             "conv_op": "torch.nn.modules.conv.Conv2d",
        #             "kernel_sizes": [
        #                 [
        #                     3,
        #                     3
        #                 ],
        #                 [
        #                     3,
        #                     3
        #                 ],
        #                 [
        #                     3,
        #                     3
        #                 ],
        #                 [
        #                     3,
        #                     3
        #                 ],
        #                 [
        #                     3,
        #                     3
        #                 ],
        #                 [
        #                     3,
        #                     3
        #                 ],
        #                 [
        #                     3,
        #                     3
        #                 ],
        #                 [
        #                     3,
        #                     3
        #                 ]
        #             ],
        #             "strides": [
        #                 [
        #                     1,
        #                     1
        #                 ],
        #                 [
        #                     2,
        #                     2
        #                 ],
        #                 [
        #                     2,
        #                     2
        #                 ],
        #                 [
        #                     2,
        #                     2
        #                 ],
        #                 [
        #                     2,
        #                     2
        #                 ],
        #                 [
        #                     2,
        #                     2
        #                 ],
        #                 [
        #                     2,
        #                     2
        #                 ],
        #                 [
        #                     2,
        #                     2
        #                 ]
        #             ],
        #             "n_blocks_per_stage": [
        #                 1,
        #                 3,
        #                 4,
        #                 6,
        #                 6,
        #                 6,
        #                 6,
        #                 6
        #             ],
        #             "n_conv_per_stage_decoder": [
        #                 1,
        #                 1,
        #                 1,
        #                 1,
        #                 1,
        #                 1,
        #                 1
        #             ],
        #             "conv_bias": true,
        #             "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
        #             "norm_op_kwargs": {
        #                 "eps": 1e-05,
        #                 "affine": true
        #             },
        #             "dropout_op": null,
        #             "dropout_op_kwargs": null,
        #             "nonlin": "torch.nn.LeakyReLU",
        #             "nonlin_kwargs": {
        #                 "inplace": true
        #             }
        #         },
        #         "_kw_requires_import": [
        #             "conv_op",
        #             "norm_op",
        #             "dropout_op",
        #             "nonlin"
        #         ]
        #     },
        #     "batch_dice": true
        # },
