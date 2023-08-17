# Copyright (c) SenseTime.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Swin: https://github.com/microsoft/Swin-Transformer
# timm: https://github.com/rwightman/pytorch-image-models
# MAE:  https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils_pretrain.pos_embed import get_2d_sincos_pos_embed
from utils_pretrain.low_freq_generator import LowFreqTargetGenerator

from swin_transformer_v2 import SwinTransformerV2

from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block


class SwinTransformerV2ForMixMIM(SwinTransformerV2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

    def forward(self, x, mask_s1, mask_s2, mask_s3, mask_s4):
        x = self.patch_embed(x)

        assert mask_s1 is not None

        # mask_tokens = self.mask_token.expand(B, L, -1)
        # w = mask_s1.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        # x = x * (1. - w) + mask_tokens * w
        x = x * (1. - mask_s1) + x.flip(0) * mask_s1
        
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                x = layer(x)
                # x = layer(x, attn_mask=mask_s1)
            elif idx == 1:
                x = layer(x)
                # x = layer(x, attn_mask=mask_s2)
            elif idx == 2:
                x = layer(x)
                # x = layer(x, attn_mask=mask_s3)
            elif idx == 3:
                x = layer(x)
                # x = layer(x, attn_mask=mask_s4)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)

        return x


class MixMIM(nn.Module):
    def __init__(self,
                img_size=128,
                encoder_stride=32,
                low_freq_target=False,
                norm_pix_loss=True, 
                range_mask_ratio=0.0, 
                norm_layer=nn.LayerNorm,
                decoder_dim=512, 
                decoder_depth=8, 
                decoder_num_heads=16,
                **kwargs
                 ):
        super().__init__()

        self.img_size = img_size

        # self.encoder = encoder
        self.encoder_stride = encoder_stride

        model_kwargs = dict(
            img_size=img_size,
            **kwargs
        )

        self.encoder = SwinTransformerV2ForMixMIM(**model_kwargs)

        # reconstruction args
        self.low_freq_target = low_freq_target
        self.norm_pix_loss = norm_pix_loss
        self.range_mask_ratio = range_mask_ratio

        self.low_freq_generator = LowFreqTargetGenerator(img_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

        # decoder args
        self.decoder_dim = decoder_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads

        out_num_patches = (img_size // self.encoder_stride) ** 2
        self.out_num_patches = out_num_patches
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, out_num_patches, decoder_dim), requires_grad=False)

        self.decoder_embed = nn.Linear(self.encoder.num_features, decoder_dim)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_dim, decoder_num_heads, self.encoder.mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_dim)
        self.decoder_pred = nn.Linear(
            decoder_dim,
            self.encoder_stride ** 2 * 3
        )

        # self.in_chans = self.encoder.in_chans
        # self.patch_size = self.encoder.patch_size

    def random_masking(self, x, mask_ratio):
        B, C, H, W = x.shape
        out_H = H // self.encoder_stride
        out_W = W // self.encoder_stride
        s3_H, s3_W = out_H * 2, out_W * 2
        s2_H, s2_W = out_H * 4, out_W * 4
        s1_H, s1_W = out_H * 8, out_W * 8

        seq_l = out_H * out_W
        # use a shared mask for a batch images
        mask = torch.zeros([1, 1, seq_l], device=x.device)

        mask_ratio = mask_ratio + random.uniform(0.0, self.range_mask_ratio)
        noise = torch.rand(1, 1, seq_l, device=x.device)  # noise in [0, 1]
        # ascend: small is keep, large is remove
        mask_idx = torch.argsort(noise, dim=2)[:, :, :int(seq_l * mask_ratio)]
        mask.scatter_(2, mask_idx, 1)
        mask = mask.reshape(1, 1, out_H, out_W)
        mask_s1 = torch.nn.functional.interpolate(mask, size=(s1_H, s1_W), mode='nearest')
        mask_s2 = torch.nn.functional.interpolate(mask, size=(s2_H, s2_W), mode='nearest')
        mask_s3 = torch.nn.functional.interpolate(mask, size=(s3_H, s3_W), mode='nearest')

        mask = mask.reshape(1, out_H * out_W, 1).contiguous()
        mask_s1 = mask_s1.reshape(1, s1_H * s1_W, 1).contiguous()
        mask_s2 = mask_s2.reshape(1, s2_H * s2_W, 1).contiguous()
        mask_s3 = mask_s3.reshape(1, s3_H * s3_W, 1).contiguous()

        return mask_s1, mask_s2, mask_s3, mask

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.encoder_stride
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.encoder_stride
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_decoder(self, x, mask):
        # embed tokens
        x = self.decoder_embed(x)
        B, L, C = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)
        x1 = x * (1 - mask) + mask_tokens * mask
        x2 = x * mask + mask_tokens * (1 - mask)
        x = torch.cat([x1, x2], dim=0)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for idx, blk in enumerate(self.decoder_blocks):
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, x, x_rec, mask):
        B, L, C = x_rec.shape

        # unmix tokens
        x1_rec = x_rec[:B//2]
        x2_rec = x_rec[B//2:]

        if self.low_freq_target:
            x = self.low_freq_generator(x)

        target = self.patchify(x)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        unmix_x_rec = x1_rec * mask + x2_rec.flip(0) * (1 - mask)
        loss_rec = (unmix_x_rec - target) ** 2
        loss_rec = loss_rec.mean()

        return loss_rec

    def forward(self, x, mask_ratio=0.5):

        B, C, H, W = x.shape

        mask_s1, mask_s2, mask_s3, mask_s4 = self.random_masking(x, mask_ratio)
        # mask_s1, mask_s2, mask_s3, mask_s4 = mask_s1.cuda(), mask_s2.cuda(), mask_s3.cuda(), mask_s4.cuda()
        z = self.encoder(x, mask_s1, mask_s2, mask_s3, mask_s4)
        z = z.reshape(B, -1, self.encoder.num_features).contiguous()
        x_rec = self.forward_decoder(z, mask_s4)
        loss = self.forward_loss(x, x_rec, mask_s4)
        return loss, x_rec, mask_s4
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}
    
    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


# @register_model
def mixmae_cd_tiny_swin_v2(**kwargs):
    mixmae_cd_default_args = dict(
        img_size=128,
        encoder_stride=32,
        norm_pix_loss=True,
        range_mask_ratio=0.0,
        norm_layer=nn.LayerNorm,
        decoder_dim=512, 
        decoder_depth=2, 
        decoder_num_heads=16,
        patch_size=4,
        in_chans=3,
        num_classes=0,
        embed_dim=32,
        depths=(2, 2, 4, 2),
        num_heads=(2, 2, 2, 2),
        window_size=8,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.0,
        ape=False,
        patch_norm=True,
        use_checkpoint=None,
    )

    mixmae_cd_default_args.update(**kwargs)
    model = MixMIM(**mixmae_cd_default_args)

    return model