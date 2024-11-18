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
from .vit_decoder import ViTDecoder
from .conv_decoder import ConvDecoder
from .sam_decoder import SAMDecoder

import torch.nn.functional as F
import numpy as np 




class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_frames=16,
        t_patch_size=2,
        patch_embed=video_vit.PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=True,
        sep_pos_embed_decoder=False,
        trunc_init=False,
        cls_embed=False,
        pred_t_dim=16,
        num_classes=1,
        use_lora=0, # 0 for not use lora
        decoder_type='vit',# conv, conv_up , dpt
        deep_supervision=False,
        **kwargs,
    ):
        super().__init__()
        # cls_embed = True
        assert decoder_type in ['vit', 'conv', 'conv_up', 'dpt', 'mask'], "decoder_type can be 'vit', 'conv', 'conv_up', 'dpt', '2d_conv'"
        
        cls_embed_decoder = False 
        self.cls_embed_decoder = cls_embed_decoder

        
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
        self.patch_size = patch_size
        self.patch_info = None

        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )

        num_patches = self.patch_embed.num_patches
        print("num_patches", num_patches)
        input_size = self.patch_embed.input_size
        self.input_size = input_size
        self.output_size = [num_frames,img_size, img_size]
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.dpt_feat_layer = [5,11,17]
        # print("self.input_size", self.input_size) #(8, 14, 14)
 
        
        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )

            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                video_vit.Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    use_lora=use_lora,
                )
                for i in range(depth)
            ]
        )

        self.decoder_type = decoder_type
        self.norm = norm_layer(embed_dim)
        
        if decoder_type=='dpt':
            self.use_readout = 'ignore'
            self.scratch = nn.Module()
            in_shape = [128,256,512,512]
            out_shape1, out_shape2, out_shape3, out_shape4 = 256,256,256,256
            self.scratch.layer1_rn = nn.Conv3d(
                in_shape[0],
                out_shape1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.act_postprocess1 = nn.Sequential(
                nn.Conv3d(
                in_channels=embed_dim,
                out_channels=in_shape[0],
                kernel_size=1,
                stride=1,
                padding=0,
                ),
                nn.ConvTranspose3d(
                    in_channels=in_shape[0],
                    out_channels=in_shape[0],
                    kernel_size=4,
                    stride=(2,4,4),
                    padding=(1,0,0),
                    bias=True,
                    dilation=1,
                ))           

            self.scratch.layer2_rn = nn.Conv3d(
                in_shape[1],
                out_shape2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )


            self.act_postprocess2 = nn.Sequential(
                nn.Conv3d(
                in_channels=embed_dim,
                out_channels=in_shape[1],
                kernel_size=1,
                stride=1,
                padding=0,
                ),
                nn.ConvTranspose3d(
                    in_channels=in_shape[1],
                    out_channels=in_shape[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=True,
                    dilation=1,
                ))    
            

            self.scratch.layer3_rn = nn.Conv3d(
                in_shape[2],
                out_shape3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )


            self.act_postprocess3 = nn.Sequential(
                nn.Conv3d(
                in_channels=embed_dim,
                out_channels=in_shape[2],
                kernel_size=1,
                stride=1,
                padding=0,
                ))
            
            self.scratch.layer4_rn = nn.Conv3d(
                in_shape[3],
                out_shape4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )

            self.act_postprocess4 = nn.Sequential(
                nn.Conv3d(
                    in_channels=embed_dim,
                    out_channels=in_shape[3],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                nn.Conv3d(
                    in_channels=in_shape[3],
                    out_channels=in_shape[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            )

            self.scratch.refinenet1 = _make_fusion_block(256, feature_align_shape=(self.num_frames, self.output_size[1]//2, self.output_size[2]//2 ))
            self.scratch.refinenet2 = _make_fusion_block(256, feature_align_shape=(self.num_frames, self.output_size[1]//4, self.output_size[2]//4))
            self.scratch.refinenet3 = _make_fusion_block(256, feature_align_shape=(self.num_frames, self.output_size[1]//8, self.output_size[2]//8))
            self.scratch.refinenet4 = _make_fusion_block(256, feature_align_shape=((self.num_frames//2, self.output_size[1]//16, self.output_size[2]//16)))
            
            self.head = nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
                # nn.BatchNorm3d(256),
                LayerNorm3d(256),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv3d(256, num_classes, kernel_size=1),
            )
            # self.auxlayer = nn.Sequential(
            #     nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU(True),
            #     nn.Dropout(0.1, False),
            #     nn.Conv3d(256, num_classes, kernel_size=1),
            # )

        elif decoder_type == 'vit':
            self.decoder = ViTDecoder(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                decoder_embed_dim=decoder_embed_dim,
                decoder_depth=decoder_depth,
                decoder_num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                num_frames=num_frames,
                t_patch_size=t_patch_size,
                no_qkv_bias=no_qkv_bias,
                sep_pos_embed=sep_pos_embed,
                sep_pos_embed_decoder=sep_pos_embed_decoder,
                trunc_init=trunc_init,
                pred_t_dim=pred_t_dim,
                cls_embed=cls_embed,
                use_lora=use_lora,
                num_classes=num_classes,
                num_patches=num_patches,
                input_size=input_size
            )

        elif self.decoder_type == 'conv':
            self.decoder = ConvDecoder(
                num_classes=num_classes,
                img_size=img_size,
                embed_dim=embed_dim,
                in_chans=in_chans,
                num_frames=num_frames,
                input_size=input_size,
                deep_supervision=deep_supervision,
            )

        elif self.decoder_type == 'mask':
            self.decoder = SAMDecoder(
                img_size=img_size,
                embed_dim=embed_dim,
                norm_layer=norm_layer,
                num_classes=num_classes,
                input_size=input_size,
                num_frames=num_frames
            )



        else:
            raise NotImplementedError


        self.initialize_weights()
        print("model initialized")
        self.use_lora = use_lora



    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

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


    def forward_encoder(self, x):
        # ([2, 3, 16, 224, 224])
        #  B, C, T, H, W 
        # print("encoder sample_x", x.size())
        multi_scale_feat = []
        multi_scale_feat.append(x)
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C) # combine temporal and spatial together
       
        x = x.view(N, -1, C)
        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)

            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        pos_embed = pos_embed.to(x.device)
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        
        for layer_i, blk in enumerate(self.blocks):
            x = blk(x)
            if layer_i in self.dpt_feat_layer:
                multi_scale_feat.append(x)


        x = self.norm(x)
        multi_scale_feat.append(x)


        return x, multi_scale_feat

    def forward_decoder(self, x, multi_scale_feat=None, t = None):
        # x (4, 1568, 1024)
        # print("sample_t", t.size())

        # N = x.shape[0]
        # T = self.patch_embed.t_grid_size
        # H = W = self.patch_embed.grid_size

        # embed tokens
        # C = x.shape[-1]

        if self.decoder_type == 'vit':
            x = self.decoder(x, patch_info=self.patch_info)

        elif self.decoder_type == 'conv':
            x = self.decoder(multi_scale_feat)

        elif self.decoder_type == 'mask':
            x = self.decoder(x)


        # elif self.decoder_type == 'dpt':
        #     layer_1, layer_2, layer_3, layer_4 = multi_scale_feat
        #     # transpose to B, C, T, H, W
        #     layer1 = self.act_postprocess1(self.convert_to_3d_tensor(layer_1)) # ([2, 128, 16*s, 56, 56])  # ([2, 128, 64, 28, 28])
        #     layer2 = self.act_postprocess2(self.convert_to_3d_tensor(layer_2)) # ([2, 256, 16*s, 28, 28])  # ([2, 256, 32, 14, 14])
        #     layer3 = self.act_postprocess3(self.convert_to_3d_tensor(layer_3)) # ([2, 512, 8*s, 14, 14])   # ([2, 512, 16, 7, 7])
        #     layer4 = self.act_postprocess4(self.convert_to_3d_tensor(layer_4)) # ([2, 512, 4*s, 7, 7])     # ([2, 512, 8, 4, 4])


        #     layer_1_rn = self.scratch.layer1_rn(layer1) # ([2, 256, 16*s, 56, 56])
        #     layer_2_rn = self.scratch.layer2_rn(layer2) # ([2, 256, 16*s, 28, 28])
        #     layer_3_rn = self.scratch.layer3_rn(layer3) # ([2, 256, 8*s, 14, 14])
        #     layer_4_rn = self.scratch.layer4_rn(layer4) # ([2, 256, 4*s, 7, 7])


        #     path_4 = self.scratch.refinenet4(layer_4_rn) # ([2, 256, 8, 14, 14])  # ([2, 256, 32, 8, 8])
        #     path_3 = self.scratch.refinenet3(path_4, layer_3_rn) # ([2, 256, 8, 28, 28])
        #     path_2 = self.scratch.refinenet2(path_3, layer_2_rn) # ([2, 256, 16, 56, 56])
        #     path_1 = self.scratch.refinenet1(path_2, layer_1_rn) # ([2, 256, 16, 112, 112])
        #     x = self.head(path_1)
        #     x = F.interpolate(
        #             x,
        #             size=self.output_size,
        #             mode="area",
        #         )
        else:
            raise NotImplementedError
        return x

    def convert_to_3d_tensor(self, x):
        # TODO use_readout
        if self.use_readout == 'ignore' and self.cls_embed :
            x = x[:, 1:]
        N = x.shape[0]
        C = x.shape[-1]
        # print("x size", x.size())
        x = x.view([N, self.input_size[0],self.input_size[1], self.input_size[2], C]) # B, 8, 14, 14, 512
        x = x.permute(0, 4, 1, 2, 3) # B, 1024, 8, 14, 14 
        return x

 
    def forward(self, imgs):
        input_dim = imgs.shape[1]
        if input_dim == 1:
            imgs = imgs.repeat(1,3,1,1,1)
        # print("====================================")
        # print("imgs", imgs.size())
        _ = self.patchify(imgs)
        latent, multi_scale_feat = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent,multi_scale_feat=multi_scale_feat)
        return pred

    def forward_visualize(self, imgs, seg, t=None, visualize=False):
        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, t)  # [N, L, p*p*3] ([2, 1568, 22016])
       
        # loss, pred_class = self.forward_loss(seg, pred, visualize)
        _ = self.patchify(imgs)
        pred_class = self.unpatchify(pred)

        pred_class = torch.sigmoid(pred_class)
   
        N, T, H, W, p, u, t, h, w = self.patch_info
        assert imgs.shape[2]==seg.shape[2]
        divide_slices = torch.linspace(0,imgs.shape[2] - 1,self.pred_t_dim,).long().to(imgs.device)

        imgs_ = imgs
        segmentation_gt = seg

        segmentation_pred = torch.argmax(pred_class, dim=1, keepdim=True)
        
        segmentation_gt = segmentation_gt.repeat(1,3,1,1,1)
        segmentation_pred = segmentation_pred.repeat(1,3,1,1,1)

        comparison = torch.stack([imgs_, segmentation_gt, segmentation_pred], dim=1)
        return None, pred, seg, comparison
        



def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
