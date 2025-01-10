# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath # ,  Mlp

import loralib as lora
from functools import partial


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_lora=0, # 0: False, 1: True, 2: aggressive lora

    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        # linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        if use_lora == 0 or use_lora==False or use_lora==1:
            self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
            self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        elif use_lora == 2:
            self.fc1 = lora.Linear(in_features, hidden_features, r=32, lora_alpha=32)
            self.fc2 = lora.Linear(hidden_features, out_features, r=32, lora_alpha=32)
        # self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        # self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x





# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x):
#         return self.net(x)



# class LayerNorm3d(nn.Module):
#     def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(num_channels))
#         self.bias = nn.Parameter(torch.zeros(num_channels))
#         self.eps = eps

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x [B, C, H, W, D]
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
#         return x
    

# class GlobalAttnEncoder(nn.Module):
#     """Transformer Model Encoder for sequence to sequence translation."""

#     def __init__(
#         self,
#         pos_enc: int, # -1 for no pos encoding, 0 for learnable, 1 for 1d, 2 for 3d
#         input_size: Tuple[int, int, int],
#         # seq_length: int,
#         num_layers: int,
#         num_heads: int,
#         hidden_dim: int,
#         mlp_dim: int,
#         dropout: float,
#         attention_dropout: float,
#         norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#     ):
#         super().__init__()
#         # Note that batch_size is on the first dim because
#         # we have batch_first=True in nn.MultiAttention() by default

#         self.pos_enc = pos_enc
#         if pos_enc == 0:
#             seq_length = input_size[0] * input_size[1] * input_size[2]
#             self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  
#         elif pos_enc == -1:
#             self.pos_embedding = None
#         elif pos_enc == 1:
#             self.pos_embedding = PositionalEncoding1D(hidden_dim)
#         elif pos_enc == 2:
#             self.pos_embedding = PositionalEncoding3D(hidden_dim)

#         self.dropout = nn.Dropout(dropout)
#         layers: OrderedDict[str, nn.Module] = OrderedDict()
#         for i in range(num_layers):
#             layers[f"encoder_layer_{i}"] = EncoderBlock(
#                 num_heads,
#                 hidden_dim,
#                 mlp_dim,
#                 dropout,
#                 attention_dropout,
#                 norm_layer,
#             )
#         self.layers = nn.Sequential(layers)
#         self.ln = norm_layer(hidden_dim)

#     def forward(self, input: torch.Tensor):
#         # input shape torch.Size([2, 32, 8, 112, 112])
#         B, C, H, W, D = input.shape

#         if self.pos_enc == 2:
#             pos_embed = self.pos_embedding(input)
#             input = input + pos_embed
#         input = input.view(B, C, -1)
#         input = input.permute(0, 2, 1)

#         torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

#         if self.pos_enc == 0:
#             input = input + self.pos_embedding
#         elif self.pos_enc == 1:
#             input = input + self.pos_embedding(input)
#         elif self.pos_enc == -1:
#             pass

#         output = self.ln(self.layers(self.dropout(input)))

#         output = output.permute(0, 2, 1)
#         output = output.view(B, C, H, W, D)

#         return output
