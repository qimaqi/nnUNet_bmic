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

import torch.nn.functional as F
import numpy as np 



class SAMDecoder(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        embed_dim=1024,
        norm_layer=nn.LayerNorm,
        num_frames=16,
        num_classes=1,
        input_size=(8, 14, 14),
        **kwargs,
    ):
        super().__init__()
        # cls_embed = True
        self.input_size = input_size
       


        from .sam_modeling import MaskDecoder3D
        from .sam_modeling import PromptEncoder3D


        self.norm = norm_layer(embed_dim)
        self.decoder_embed_mask = nn.Linear(embed_dim, 384, bias=True)

        self.mask_decoder = MaskDecoder3D(
        num_multimask_outputs=num_classes,
        transformer_dim=384,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        )
        self.prompt_encoder=PromptEncoder3D(
        embed_dim=384,
        image_embedding_size=(self.input_size[0], self.input_size[1], self.input_size[2]),
        input_image_size=(num_frames,img_size, img_size),
        mask_in_chans=16,
        )


    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, C, T, H, W = imgs.shape
        p = self.patch_size #self.patch_embed.patch_size[0]
        u = self.t_pred_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, C, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * C))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

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

    def convert_3d_to_2d_tensor(self, x):
        N,C, D, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4) # B, 1024, 8, 14, 14 
        x = x.reshape([N*D, C, H, W])
        return x

    def convert_2d_to_3d_tensor(self, x, N):
        ND, C, H, W = x.size()
        D = ND // N 
        x = x.reshape([N, D, C, H, W])
        x = x.permute(0, 2, 1, 3, 4)
        return x


    def forward(self, x):
        # x (4, 1568, 1024)
        # print("sample_t", t.size())


        x = self.decoder_embed_mask(x) # (4, 1568, 512)
        N = x.shape[0]
        x = x.permute(0, 2, 1).view(N,384,self.input_size[0],self.input_size[1], self.input_size[2]) # (4, 512, 8, 14, 14)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        ) 
        # print("sparse_embeddings", sparse_embeddings.size())  # ([1, 0, 384])
        # print("dense_embeddings", dense_embeddings.size())    # [1, 384, 8, 14, 14])
        multimask_output = True
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=x,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        x = low_res_masks

        return x
