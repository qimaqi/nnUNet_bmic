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
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import _load_weights
from timm.models import create_model
import math
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    # from mamba_ssm.ops.triton.layernorm_gated import RMSNorm, LayerNorm
    # from mamba_ssm.ops.triton.layernorm import layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


from .videomae import video_vit
from .videomae.video_vit import LayerNorm3d
# from .videomae.logging import master_print as print
from .vit_decoder import ViTDecoder
from .conv_decoder import ConvDecoder
from .sam_decoder import SAMDecoder

import torch.nn.functional as F
import numpy as np 

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=True, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)



# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class VisionMamba_Seg(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224, 
        patch_size=16, 
        depth=32, 
        embed_dim=576, 
        in_chans=3, 
        num_classes=3,
        drop_path_rate=0.,
        ssm_cfg=None, 
        norm_epsilon=1e-5, 
        initializer_cfg=None,
        fused_add_norm=True,
        rms_norm=True, 
        residual_in_fp32=True,
        bimamba=True,
        # video
        kernel_size=2, 
        num_frames=16, 
        device=None,
        dtype=None,
        # checkpoint
        use_checkpoint=False,
        checkpoint_num=0,
        mask_ratio=0.8,
        t_pos_embed=False,
        pred_t_dim=16,
        decoder_type='vit', # vit, conv, conv_up, 
        deep_supervision=False,
        **kwargs,
    ):  

        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.decoder_type = decoder_type
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')
   
        self.depth = depth
        self.encoder_depth = depth 
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_frames = num_frames
        self.kernel_size = kernel_size
        self.patch_info = None
        self.dpt_feat_layer = [7,15,23]


        # pretrain parameters
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=kernel_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches


        self.input_size = [num_frames // kernel_size, img_size // patch_size, img_size // patch_size]
        self.output_size = [num_frames,img_size, img_size]



        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1 , self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)
        
        if decoder_type == 'vit':
            self.decoder = ViTDecoder(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                decoder_embed_dim=embed_dim,
                decoder_depth=8,
                decoder_num_heads=16,
                mlp_ratio=4.0,
                norm_layer=nn.LayerNorm,
                num_frames=num_frames,
                t_patch_size=kernel_size,
                no_qkv_bias=False,
                sep_pos_embed=True,
                sep_pos_embed_decoder=False,
                pred_t_dim=pred_t_dim,
                cls_embed=False,
                use_lora=0,
                num_classes=num_classes,
                num_patches=num_patches*num_frames // kernel_size,
                input_size=self.input_size
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


        print("model initialized")
        # original init
        self.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )



    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)



    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, C, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.kernel_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, C, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * C))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x


    def forward_encoder(self, x):
        multi_scale_feat = []
        multi_scale_feat.append(x)
        # x ([4, 3, 64, 224, 224])
        x = self.patch_embed(x)
        # [4, 576, 32, 14, 14])
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B ,T * H * W, C)

        pos_embed = self.pos_embed[:,1:].repeat(1,T,1 )  + torch.repeat_interleave(self.temporal_pos_embedding, H*W, dim=1)
        
        pos_embed = pos_embed.expand(x.shape[0], -1, -1)

        x_vis = x.view([B, -1, C]) + pos_embed
        x_mamba_vis = []
        

        residual = None
        hidden_states = x_vis

        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=None,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=None
                )
            if idx in self.dpt_feat_layer:
                multi_scale_feat.append(hidden_states)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn 
            # if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                eps=self.norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        x = hidden_states 
        multi_scale_feat.append(x)
        return x , multi_scale_feat


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

        _ = self.patchify(imgs)
        latent, multi_scale_feat = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent,multi_scale_feat=multi_scale_feat)
        return pred

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs) # , bimamba=bimamba
    norm_cls = nn.LayerNorm #partial(nn.LayerNorm if ((not rms_norm) or (RMSNorm==None)) else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block