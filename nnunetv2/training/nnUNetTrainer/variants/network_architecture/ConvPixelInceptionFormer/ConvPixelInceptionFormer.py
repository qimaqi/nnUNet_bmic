# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from https://github.com/Project-MONAI/MONAI/blob/13b96aedc48ad2da16149490b06a1a6bd8361335/monai/networks/nets/basic_unet.py#L178

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional
import json 
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

import torch
import torch.nn as nn
from .basic_unet import BasicUnet_Encoder2Layer, BasicUnet_Decoder2Layer
from .linear_inception import ConvPixelInceptionEncoder_2Layer, ConvPixelInceptionDecoder_2Layer, ConvPixelInceptionConvEncoder_2Layer, ConvPixelInceptionConvDecoder_2Layer
from typing import Union, Type, List, Tuple, Dict, Any, Callable
import loralib as lora

# torch.autograd.detect_anomaly(check_nan=True)


# create also a standard Attention block
# from huggingface transformer


# class BasicUnet_Decoder2Layer(nn.Module):
#     def __init__(
#         self,
#         spatial_dims: int = 3,
#         in_channels: int = 1,
#         out_channels: int = 2,
#         features: Sequence[int] = (32, 32, 64, 32),
#         patch_size: Sequence[int] = (20, 256, 224),
#         act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
#         norm: str | tuple = ("GROUP", {"num_groups": 1, "affine": False}),  #("instance", {"affine": True}),
#         bias: bool = True,
#         dropout: float | tuple = 0.0,
#         upsample: str = "deconv",
#         attn_dict: dict = None,
#     ):
#         super().__init__()
#         fea = ensure_tuple_rep(features, 4)
#         print(f"BasicUNet features: {fea}.")

#         self.upcat_2 = UpCat(
#             spatial_dims=spatial_dims, in_chns=fea[2], 
#             cat_chns=fea[1], out_chns=fea[1], 
#             act=act, 
#             norm=norm, 
#             bias=bias, 
#             dropout=dropout, 
#             upsample=upsample)
        

#         self.upcat_1 = UpCat(
#             spatial_dims=spatial_dims, 
#             in_chns=fea[1], cat_chns=fea[0], out_chns=fea[-1], act=act, norm=norm, bias=bias, dropout=dropout, upsample=upsample, halves=False)

#         feature_shape_1 = patch_size
#         feature_shape_2 = [patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]


#         if attn_dict is not None:
#             if attn_dict['attn_type'] == 'global':
#                 self.attn_block1 = GlobalAttnEncoder(
#                     pos_enc=attn_dict['pos_enc'], input_size=feature_shape_1, num_layers=attn_dict['stack_num'][0], num_heads=attn_dict['head_dims'][0], hidden_dim=fea[-1], mlp_dim=fea[-1], dropout=attn_dict['dropout'], attention_dropout=attn_dict['attention_dropout'],
#                 )

#                 if attn_dict['pos_enc'] not in [10,20]:
#                     rest_pos_enc = -1 
#                 else:
#                     rest_pos_enc = attn_dict['pos_enc'] // 10

#                 self.attn_block2 = GlobalAttnEncoder(
#                     pos_enc=rest_pos_enc , input_size=feature_shape_2,
#                     num_layers=attn_dict['stack_num'][1], num_heads=attn_dict['head_dims'][1], hidden_dim=fea[1], mlp_dim=fea[1],dropout=attn_dict['dropout'], attention_dropout=attn_dict['attention_dropout'],
#                 )


#             elif attn_dict['attn_type'] == 'window':
#                 self.attn_block1 = BasicLayer(
#                     dim = fea[-1],
#                     depth = attn_dict['stack_num'][0],
#                     num_heads = attn_dict['head_dims'][0],
#                     window_size = attn_dict['window_size'],
#                     downsample=attn_dict['downsample'][0],
#                 )

#                 self.attn_block2 = BasicLayer(
#                     dim = fea[1],
#                     depth = attn_dict['stack_num'][1],
#                     num_heads = attn_dict['head_dims'][1],
#                     window_size = attn_dict['window_size'],
#                     downsample=attn_dict['downsample'][1],
#                 )
            

#             else:
#                 self.attn_block1 = nn.Identity()
#                 self.attn_block2 = nn.Identity()


#         self.final_conv = Conv["conv", spatial_dims](fea[-1], out_channels, kernel_size=1)

#     def forward(self, x0, x1, x2):

#         u2 = self.upcat_2(x2, x1)
#         u2 = self.attn_block2(u2)
#         u1 = self.upcat_1(u2, x0)
#         u1 = self.attn_block1(u1)
#         # u2 shape torch.Size([2, 32, 10, 128, 112])
#         # u1 shape torch.Size([2, 32, 20, 256, 224])
#         logits = self.final_conv(u1)
#         return logits
    

# class BasicUnet_Encoder2Layer(nn.Module):
#     def __init__(
#         self,
#         spatial_dims: int = 3,
#         in_channels: int = 1,
#         out_channels: int = 2,
#         patch_size: Sequence[int] = (20, 256, 224),
#         features: Sequence[int] = (32, 32, 64, 32),
#         act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
#         norm: str | tuple = ("instance", {"affine": True}),
#         bias: bool = True,
#         dropout: float | tuple = 0.0,
#         upsample: str = "deconv",
#         # now we add attention related arguments
#         attn_dict: dict = None,
#     ):
#         super().__init__()
#         fea = ensure_tuple_rep(features, 4)
#         print(f"BasicUNet features: {fea}.")
#         # spatial_dims: int,
#         # in_chns: int,
#         # out_chns: int,
#         # act: str | tuple,
#         # norm: str | tuple,
#         # bias: bool,
#         # dropout: float | tuple = 0.0,
#         self.conv_0 = TwoConv(spatial_dims=spatial_dims, in_chns=in_channels, out_chns=features[0], act=act, norm=norm, bias=bias, dropout=dropout)
#         self.down_1 = Down(spatial_dims=spatial_dims, in_chns=fea[0], out_chns=fea[1], act=act, norm=norm, bias=bias, dropout=dropout)
#         self.down_2 = Down(spatial_dims=spatial_dims, in_chns=fea[1], out_chns=fea[2], act=act, norm=norm, bias=bias, dropout=dropout)

#         # add also attention block
#         # calculate feature size
#         # TODO, for other noneven size, we need to pad to even size?
#         if attn_dict is not None:
#             # using attention blocks, do some sanity check
#             assert attn_dict['attn_type'] in ['global', 'window', 'linear_global', 'linear_window', 'linear_inception']
#             print("attn_dict", attn_dict)
#             # sanity check for some basic parameters
#             # for torchvision attention: seq_length(for pose encoding), num_layers, num_heads, hidden_dim, mlp_dim, dropout,
#             if attn_dict['attn_type'] == 'global':  
#                 assert len(attn_dict['head_dims']) == len(fea) - 2

#                 assert len(attn_dict['stack_num']) == len(fea) - 2

#                 assert attn_dict['pos_enc'] in [0, 1, 2, 10, 20]

#         feature_shape_1 = [patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]
#         feature_shape_2 = [patch_size[0] // 4, patch_size[1] // 4, patch_size[2] // 4]

#         if attn_dict['attn_type'] == 'global': 
#             self.attn_block1 = GlobalAttnEncoder(
#                 pos_enc=attn_dict['pos_enc'], input_size=feature_shape_1, num_layers=attn_dict['stack_num'][0], num_heads=attn_dict['head_dims'][0], hidden_dim=fea[1], mlp_dim=fea[1], attention_dropout=attn_dict['attention_dropout'], dropout=attn_dict['dropout']
#             )

#             if attn_dict['pos_enc'] not in [10,20]:
#                 rest_pos_enc = -1 
#             else:
#                 rest_pos_enc = attn_dict['pos_enc'] // 10

#             self.attn_block2 = GlobalAttnEncoder(
#                 pos_enc=rest_pos_enc, input_size=feature_shape_2, num_layers=attn_dict['stack_num'][1], num_heads=attn_dict['head_dims'][1], hidden_dim=fea[2], mlp_dim=fea[2], attention_dropout=attn_dict['attention_dropout'], dropout=attn_dict['dropout']
#             )
#         # self, 
#         # dim: int,
#         # depth: int,
#         # pos_enc: int, 
#         # input_size: Tuple[int, int, int],
#         # num_layers: int,
#         # num_heads: int,
#         # window_size: Sequence[int],
#         # downsample = PatchMerging,
#         # use_checkpoint: bool = False,
#         # dropout: float = 0.0,
#         # attn_drop: float = 0.0,
#         # norm_layer: type[LayerNorm] = nn.LayerNorm,
#         elif attn_dict['attn_type'] == 'window':
#             raise NotImplementedError
#             # self.attn_block1 = WindowAttnEncoder(
#             #     pos_enc=attn_dict['pos_enc'], input_size=feature_shape_1, num_layers=attn_dict['stack_num'][0], num_heads=attn_dict['head_dims'][0], dim=fea[1], window_size=attn_dict['window_size'], dropout=attn_dict['dropout'], depth=attn_dict['stack_num'][0], attn_drop=attn_dict['attention_dropout']
#             # )
#             # self.attn_block2 = WindowAttnEncoder(
#             #     pos_enc=attn_dict['pos_enc'], input_size=feature_shape_2, num_layers=attn_dict['stack_num'][1], num_heads=attn_dict['head_dims'][1], dim=fea[2], window_size=attn_dict['window_size'], dropout=attn_dict['dropout'], depth=attn_dict['stack_num'][1], attn_drop=attn_dict['attention_dropout']
#             # )
#         elif attn_dict['attn_type'] == 'linear_global':
#             self.attn_block1 = LinearAtnnBlock(
#                 dim=fea[1], num_heads=attn_dict['head_dims'][0], pos_enc=attn_dict['pos_enc'], qk_dim_compresion=attn_dict['qk_dim_compresion'][0], qk_nonlin=attn_dict['qk_nonlin'], qkv_func=attn_dict['qkv_func'], map_func=attn_dict['map_func'], feat_shape=feature_shape_1, num_layers=attn_dict['stack_num'][0]
#             )

#             if attn_dict['pos_enc'] not in [10,20]:
#                 rest_pos_enc = -1 
#             else:
#                 rest_pos_enc = attn_dict['pos_enc'] // 10

#             self.attn_block2 = LinearAtnnBlock(
#                 dim=fea[2], num_heads=attn_dict['head_dims'][1], pos_enc=rest_pos_enc, qk_dim_compresion=attn_dict['qk_dim_compresion'][1], qk_nonlin=attn_dict['qk_nonlin'], qkv_func=attn_dict['qkv_func'], map_func=attn_dict['map_func'], feat_shape=feature_shape_2, num_layers=attn_dict['stack_num'][1]
#             )
#         elif attn_dict['attn_type'] == 'linear_window':
#             raise NotImplementedError
#         elif attn_dict['attn_type'] == 'linear_inception':
#             self.attn_block1 = LinearInceptionAtnnBlock(
#                 dim=fea[1], num_heads=attn_dict['head_dims'][0], pos_enc=attn_dict['pos_enc'], qk_dim_compresion=attn_dict['qk_dim_compresion'][0], qk_nonlin=attn_dict['qk_nonlin'], feat_shape=feature_shape_1, num_layers=attn_dict['stack_num'][0], attn_func=LinearInceptionAttention,
#                 Inception_window_size=[[8, 56, 56], [8, 28, 28], [8, 14, 14]]
#             )

#             if attn_dict['pos_enc'] not in [10,20]:
#                 rest_pos_enc = -1 
#             else:
#                 rest_pos_enc = attn_dict['pos_enc'] // 10

#             self.attn_block2 = LinearInceptionAtnnBlock(
#                 dim=fea[2], num_heads=attn_dict['head_dims'][1], pos_enc=rest_pos_enc, qk_dim_compresion=attn_dict['qk_dim_compresion'][1], qk_nonlin=attn_dict['qk_nonlin'], feat_shape=feature_shape_2, num_layers=attn_dict['stack_num'][1], attn_func=LinearInceptionAttention,
#                 Inception_window_size=[[4, 28, 28], [4, 14, 14], [4, 6, 6]]
#             )

#             self.conv_0 = nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1)
#             # TwoConv(spatial_dims=spatial_dims, in_chns=in_channels, out_chns=features[0], act=act, norm=norm, bias=bias, dropout=dropout)
#             # nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1)
#             # TwoConv(spatial_dims=spatial_dims, in_chns=in_channels, out_chns=features[0], act=act, norm=norm, bias=bias, dropout=dropout)
#             # self.down_1 = Downsample(in_dim=features[0], out_dim=features[1])
#             # self.down_2 = Downsample(in_dim=features[1], out_dim=features[2])
#             self.initialize_weights()

#         else:
#             self.attn_block1 = nn.Identity()
#             self.attn_block2 = nn.Identity()

#     def initialize_weights(self):
#         w = self.conv_0.weight.data
#         # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))  
#         # check range of weight
#         nn.init.trunc_normal_(w, std=0.02)

#     def forward(self, x: torch.Tensor):

#         x0 = self.conv_0(x)
#         x1 = self.down_1(x0)

#         # print("x1 shape", x1.shape)
#         x1 = self.attn_block1(x1)
#         # print("x1 shape", x1.shape)
#         x2 = self.down_2(x1)
#         # print("x2 shape", x2.shape)
#         x2 = self.attn_block2(x2)



#         return x0, x1, x2

# https://github.com/YuchuanTian/U-DiT/blob/main/udit_models.py
class ConvPixelInceptionFormer2Layer(nn.Module):

    def __init__(
        self,
        spatial_dims: int = 3,
        input_channels: int = 1,
        num_classes: int = 2,
        features: Sequence[int] = (32, 32, 64, 32),
        patch_size: Sequence[int] = (20, 256, 224),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        attn_dict: dict = None,
        deep_supervision: bool = False,
        deep_supervision_scales: List[int] = None,
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 4) # [feature of encoding layer, feature of fist downsample layer, feature of second downsample layer, feature of output layer before logit]

        print("====================================")

        print("Attention config", attn_dict.keys())
        # display in json way
        print(json.dumps(attn_dict, indent=4))

        # split network here
        if attn_dict is None:
            self.encoder = BasicUnet_Encoder2Layer(
                spatial_dims=spatial_dims, in_channels=input_channels, out_channels=num_classes, features=features, act=act, norm=norm, bias=bias, dropout=dropout, upsample=upsample, 
                attn_dict=attn_dict['encoder'], patch_size=patch_size
            )
            self.decoder = BasicUnet_Decoder2Layer(
                spatial_dims=spatial_dims, in_channels=input_channels, out_channels=num_classes, features=features, act=act, norm=norm, bias=bias, dropout=dropout, upsample=upsample, attn_dict=attn_dict['decoder'], patch_size=patch_size
            )

        else:
            if attn_dict["encoder"]['attn_type'] == 'linear_inception':
                if attn_dict['encoder']['map_func'] == 'mlp':
                    self.encoder = ConvPixelInceptionEncoder_2Layer(
                        spatial_dims=spatial_dims, in_channels=input_channels, out_channels=num_classes, features=features, act=act, norm=norm, bias=bias, dropout=dropout, upsample=upsample, attn_dict=attn_dict['encoder'], patch_size=patch_size
                    )
                elif attn_dict['encoder']['map_func'] == 'conv':
                    self.encoder = ConvPixelInceptionConvEncoder_2Layer(
                        spatial_dims=spatial_dims, in_channels=input_channels, out_channels=num_classes, features=features, act=act, norm=norm, bias=bias, dropout=dropout, upsample=upsample, attn_dict=attn_dict['encoder'], patch_size=patch_size
                    )
            else:
                raise NotImplementedError

            if attn_dict["decoder"]['attn_type'] == 'linear_inception':
                if attn_dict['decoder']['map_func'] == 'mlp':
                    self.decoder = ConvPixelInceptionDecoder_2Layer(
                        spatial_dims=spatial_dims, in_channels=input_channels, out_channels=num_classes, features=features, act=act, norm=norm, bias=bias, dropout=dropout, upsample=upsample, attn_dict=attn_dict['decoder'], patch_size=patch_size
                    )
                elif attn_dict['decoder']['map_func'] == 'conv':
                    self.decoder = ConvPixelInceptionConvDecoder_2Layer(
                        spatial_dims=spatial_dims, in_channels=input_channels, out_channels=num_classes, features=features, act=act, norm=norm, bias=bias, dropout=dropout, upsample=upsample, attn_dict=attn_dict['decoder'], patch_size=patch_size
                    )
            else:
                self.decoder = BasicUnet_Decoder2Layer(
                spatial_dims=spatial_dims, in_channels=input_channels, out_channels=num_classes, features=features, act=act, norm=norm, bias=bias, dropout=dropout, upsample=upsample, attn_dict=attn_dict['decoder'], patch_size=patch_size
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `spatial_dims`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """  
        x0, x1, x2 = self.encoder(x)
        logits = self.decoder(x0, x1, x2)

        return logits


# BasicUnet = Basicunet = basicunet = BasicUNet


# if __name__ == "__main__":
#     # test the 3D integration
#     B = 1
#     num_heads = 1
#     qk_dim = 4
#     max_window = [4, 4, 4]
#     H = 4
#     W = 4
#     D = 4
#     q = torch.ones(B*num_heads, qk_dim // num_heads, H, W, D)
#     k = torch.ones(B*num_heads, qk_dim // num_heads, H, W, D)
#     v = torch.ones(B*num_heads, qk_dim // num_heads, H, W, D) * 2

#     # largest window size
#     # overpad to make meaningful padding for integral image for later dialated convolution
#     intergral_kv_map = build_3D_kv_integral_image(k, v, pad=max_window) # B, F, F, H, W, D
#     intergral_k_map = build_3D_k_integral_image(k, pad=max_window) # B, F, H, W, D

#     print("intergral_kv_map", intergral_kv_map.shape)
#     print("intergral_k_map", intergral_k_map.shape)
#     # print("intergral_kv_map[0,0,0]", intergral_kv_map[:,:,:,0,0,0])
#     # print("intergral_k_map[0,0,0]", intergral_k_map[:,:,0,0,0])
#     # print("intergral_k_map[0,0,1]", intergral_k_map[:,:,0,1,0])
#     # print("intergral_k_map[1,1,1]", intergral_k_map[:,:,1,1,1])

#     # print("intergral_k_map[0,0,0]", intergral_k_map[:,:,4,4,4])
#     # print("intergral_k_map[0,0,1]", intergral_k_map[:,:,5,5,5])
#     # print("intergral_k_map[1,1,1]", intergral_k_map[:,:,5,5,4])

#     B_k, F_kk, H_k, W_k, D_k = intergral_k_map.shape
#     intergral_k_map_reshape = intergral_k_map.reshape(B_k*F_kk, 1, H_k, W_k, D_k)
        
#     window_size = [4, 4, 4]

#     assert len(window_size) == 3
#     kernel, dialation_size = build_convolve_kernel_3D_from_window_size(window_size)
#     kernel = kernel.unsqueeze(0).unsqueeze(0)
#     # make kernel not trainable
#     kernel.requires_grad = False

#     # intergral_kv_map_padded = pad_intergral_kv_map(intergral_kv_map_reshape, max_window, window_size)
#     intergral_k_map_padded = pad_intergral_kv_map(intergral_k_map_reshape, max_window, window_size)
#     t = time.time()

#     # let calculate [0,0,0] by retriving the value from intergral_kv_map_padded
#     print("intergral_k_map_padded[:,:,0,0,0]", intergral_k_map_padded[:,:,0,0,0])
#     print("intergral_k_map_padded[:,:,0,0,1]", intergral_k_map_padded[:,:,0,0,1])
#     print("intergral_k_map_padded[:,:,1,1,1]", intergral_k_map_padded[:,:,1,1,1])
#     print("intergral_k_map_padded[:,:,-2,-2,-2]", intergral_k_map_padded[:,:,-2,-2,-2])
#     print("intergral_k_map_padded[:,:,-1,-1,-1]", intergral_k_map_padded[:,:,-1,-1,-1])
#     # intergral_k_map_padded_0 = intergral_k_map_padded[:,:,0,0,0]
#     # print("intergral_kv_map_padded", intergral_kv_map_padded.shape, intergral_kv_map_padded.min(), intergral_kv_map_padded.max())
#     # print("intergral_k_map_padded", intergral_k_map_padded.shape, intergral_k_map_padded.min(), intergral_k_map_padded.max())

#     # intergral_kv_map_padded = intergral_kv_map_padded.to(dtype=torch.float32)
#     intergral_k_map_padded = intergral_k_map_padded.to(dtype=torch.float32).to(device='cuda')
#     kernel = kernel.to(dtype=torch.float32).to(device=intergral_k_map_padded.device)
#     print("intergral_k_map_padded", intergral_k_map_padded.shape)
#     print("kernel", kernel.shape, kernel)
#     print("dialation_size", dialation_size)
#     with torch.cuda.amp.autocast(enabled=False):
#         window_k = torch.nn.functional.conv3d(input=intergral_k_map_padded, weight=kernel, dilation=tuple(dialation_size), padding=0, stride=1)
#         # window_k = torch.nn.functional.conv3d(input=intergral_k_map_padded, weight=kernel, padding=0, stride=1)

#     print("window_k", window_k.shape)
#     print("window_k", window_k[:,:,0,0,0])
#     print("window_k", window_k[:,:,0,1,0])
#     print("window_k", window_k[:,:,1,1,1])
#     print("window_k", window_k[:,:,-1,-2,-2])
#     print("window_k", window_k[:,:,-1,-1,-1])
#     # # test the model
#     # model = ConvPixelInceptionFormer2Layer(
#     #     spatial_dims=3,
#     #     input_channels=1,
#     #     num_classes=2,
#     #     features=(32, 32, 64, 32),
#     #     patch_size=(20, 256, 224),
#     #     act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
#     #     norm=("instance", {"affine": True}),
#     #     bias=True,
#     #     dropout=0.0,
#     #     upsample="deconv",
#     #     attn_dict={
#     #         'encoder': {
#     #             'attn_type': 'linear_inception',
#     #             'pos_enc': 1,
#     #             'stack_num': [1, 1],
#     #             'head_dims': [1, 1],
#     #             'qk_dim_compresion': [1.0, 1.0],
#     #             'qk_nonlin': 'softmax',
#     #             'qkv_func': 'linear',
#     #             'map_func': 'conv',
#     #             'dropout': 0.0,
#     #             'attention_dropout': 0.0,
#     #         },
#     #         'decoder': {
#     #             'attn_type': 'linear_inception',
#     #             'pos_enc': 1,
#     #             'stack_num': [1, 1],
#     #             'head_dims': [1, 1],
#     #             'qk_dim_compresion': [1.0, 1.0],
#     #             'qk_nonlin': 'softmax',
#     #             'qkv_func': 'linear',
#     #             'map_func': 'conv',
#     #             'dropout': 0.0,
#     #             'attention_dropout': 0.0,
#     #         }
#     #     }
#     # )
#     # print(model)
#     # x = torch.randn(2, 1, 20, 256, 224)
#     # y = model(x)
#     # print(y.shape)
#     # print(y.min(), y.max())