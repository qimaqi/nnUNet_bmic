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
# from .videomae.logging import master_print as print
import torch.nn.functional as F
import numpy as np 
from typing import List



class ViTDecoder(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim = 1024,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        num_frames=16,
        t_patch_size=2,
        no_qkv_bias=False,
        sep_pos_embed=False,
        sep_pos_embed_decoder=False,
        trunc_init=False,
        cls_embed=False,
        pred_t_dim=8,
        num_classes=1,
        use_lora=0, # 0 for not use lora
        num_patches=14*14*8,
        input_size: List[int] = [8, 14, 14],
        **kwargs,
    ):
        super().__init__()
        # cls_embed = True
   
        cls_embed_decoder = False 
        self.cls_embed_decoder = cls_embed_decoder
        self.deep_supervision = False


        
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.sep_pos_embed_decoder = sep_pos_embed_decoder
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim
        self.num_frames = num_frames
        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames
        self.num_classes = num_classes
        # 2 * 8 // 16
        self.t_patch_size = t_patch_size
        self.patch_info = None
 

        num_patches = num_patches
        input_size = input_size
        self.input_size = input_size
        self.output_size = [num_frames,img_size, img_size]
        self.decoder_embed_dim = decoder_embed_dim
        # print("self.input_size", self.input_size) #(8, 14, 14)


        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        if self.sep_pos_embed_decoder:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed_decoder:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed_decoder:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self._num_patches = _num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        # we do not use lora for decoder
        self.decoder_blocks = nn.ModuleList(
            [
                video_vit.Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    use_lora=False,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # TODO change to segmentation head
        self.decoder_seg = nn.Linear(
            decoder_embed_dim, # 512
            self.t_pred_patch_size * patch_size**2 * num_classes, # 16*16*3
            bias=True,
        )
        self.initialize_weights()
        print("model initialized")
        self.use_lora = use_lora

    def initialize_weights(self):
 


        if self.sep_pos_embed_decoder:
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed_decoder:
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)


        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, -1))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, -1, T, H, W))
        return imgs


    def forward(self, x, patch_info=None):
        if patch_info is not None:
            self.patch_info = patch_info

        x = self.decoder_embed(x) # (4, 1568, 512)
        if self.cls_embed_decoder:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed_decoder:
            decoder_pos_embed_spatial = self.decoder_pos_embed_spatial.repeat(
                    1, self.input_size[0], 1
                )
            
            decoder_pos_embed_temporal = torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            decoder_pos_embed = decoder_pos_embed_spatial + decoder_pos_embed_temporal

            if self.cls_embed_decoder:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]
            if self.cls_embed:
                x = x[:, 1:, :]
            
        decoder_pos_embed = decoder_pos_embed.to(x.device)
        # add pos embed
        x = x + decoder_pos_embed
        
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        #  print("final decoder embedding", x.size())
        x = self.decoder_seg(x)

        x = self.unpatchify(x) 

        # print("====================================")
        # print("pred", x.size())
        # print("================================")
        return x
    
    # def forward(self, imgs, t=None):
    #     _ = self.patchify(imgs)
    #     latent, multi_scale_feat = self.forward_encoder(imgs)
    #     pred = self.forward_decoder(latent, t=t, multi_scale_feat=multi_scale_feat)
    #     return pred

