# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import math
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp

from util.utils import off_diagonal
from util.pos_embed import get_2d_sincos_pos_embed


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_block=False):
        if attention_block == False:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        elif attention_block == True:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            return attn


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, attention_block=False):
        if attention_block == False:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x
        elif attention_block == True:
            attention_matrix = self.attn(self.norm1(x), True)
            return attention_matrix


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, num_classes=1000, norm_pix_loss=False,
                 frozen_emb=False, drop_rate=0.01):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop=drop_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop=drop_rate)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans,
                                      bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights(frozen_emb, embed_dim)

    def initialize_weights(self, frozen_emb, embed_dim):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        if isinstance(self.patch_embed, PatchEmbed) and frozen_emb:
            # xavier_uniform initialization moco v3
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            self.patch_embed.proj.weight.requires_grad = False
            self.patch_embed.proj.bias.requires_grad = False
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_views_masking(self, x, mask_ratio, r_views):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        x_masked_list = []
        mask_list = []
        # keep the first subset
        for i in range(r_views):
            ids_keep = ids_shuffle[:, i * len_keep:(i + 1) * len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            x_masked_list.append(x_masked)

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, i * len_keep:(i + 1) * len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
            mask_list.append(mask)
        x_masked = torch.cat(x_masked_list, dim=0)
        mask = torch.stack(mask_list, dim=0)

        return x_masked, mask, ids_restore

    def forward_vanilla(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_encoder(self, x, mask_ratio, r_views):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_views_masking(x, mask_ratio, r_views)
        # x, mask, ids_restore = self.random_views_masking_compare(x, mask_ratio, r_views)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_views_decoder(self, x, ids_restore, r_views):
        # embed tokens
        x = self.decoder_embed(x)

        bs, N, L = x.shape
        r = r_views
        x = x.reshape(r, bs // r, N, L)
        x_list = []
        # append mask tokens to sequence
        for i in range(r):
            mask_tokens_head = self.mask_token.repeat(x.shape[1], ids_restore.shape[1] - (N - 1) * (r - i), 1)
            mask_tokens_tail = self.mask_token.repeat(x.shape[1], ids_restore.shape[1] - (N - 1) * (i + 1), 1)
            x_ = torch.cat([mask_tokens_head, x[i][:, 1:, :], mask_tokens_tail], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, L))  # unshuffle
            x_ = torch.cat([x[i][:, :1, :], x_], dim=1)  # append cls token
            x_list.append(x_)
        x_views = torch.cat(x_list, dim=0)
        # add pos embed
        x_views = x_views + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x_views = blk(x_views)
        x_views = self.decoder_norm(x_views)
        decode_cls = x_views[..., 0, :].reshape(r, bs // r, -1)
        x_views = x_views[..., 1:, :]
        x_views = self.decoder_pred(x_views)
        # remove cls token
        decode_rec = x_views.reshape(r, bs // r, ids_restore.shape[1], self.decoder_pred.out_features)
        return decode_cls, decode_rec

    def forward_loss(self, imgs, pred, mask, loss_type='mse'):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        if loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred, target, reduction='none', beta=2.0)
        else:
            loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, img, args):
        r = args.r_views
        mask_ratio = args.mask_ratio

        # This also has the class token
        latent_vanilla = self.forward_vanilla(img)
        #
        cls_vanilla = latent_vanilla[:, 0, :]
        predicted_class = self.head(cls_vanilla)  # Class predictions by the network
        if args.lambda_weight < 1:
            # This has the class token appended to it
            latent, mask, ids_restore = self.forward_encoder(img, mask_ratio, r)
            # decoder has class token to align
            decode_cls, decode_rec = self.forward_views_decoder(latent, ids_restore, r)
            loss_rec = 0
            for i, pred_rec in enumerate(decode_rec):
                loss_rec += self.forward_loss(img, pred_rec, mask[i], loss_type="mse")
            loss_rec /= r
        else:
            loss_rec = 0

        # consistency_loss
        if args.sim_beta > 0:
            bs, _, L = latent.shape
            cls_rec_avg = decode_cls.clone().detach().mean(dim=0)
            cls_rec_avg = (cls_rec_avg - cls_rec_avg.mean(dim=0)) / cls_rec_avg.std(dim=0)
            ct_loss = 0
            for z_1 in decode_cls:
                z_1 = (z_1 - z_1.mean(dim=0)) / z_1.std(dim=0)
                c = z_1.T @ cls_rec_avg
                c.div_((bs // r))
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).add_(1).pow_(2).sum()
                ct_loss += (on_diag + 0.005 * off_diag)
            sim_loss = ct_loss / r
        else:
            sim_loss = 0

        return loss_rec, sim_loss, predicted_class

    def forward_test(self, img):
        output = self.forward_vanilla(img)
        class_token = output[:, 0, :]
        predicted_class = self.head(class_token)
        return predicted_class

    def forward_attentions(self, img):
        x = self.patch_embed(img)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        attention_matrixs = []
        # apply Transformer blocks
        for blk in self.blocks:
            attention_matrix = blk(x, True)
            x = blk(x)
            attention_matrixs.append(attention_matrix)
        return attention_matrixs

    def forward_mask_image(self, image, mask_ratio=0.75, r=4):
        latent, mask, ids_restore = self.forward_encoder(image, mask_ratio, r)
        # decoder has class token to align
        decode_cls, decode_rec = self.forward_views_decoder(latent, ids_restore, r)
        return decode_cls, decode_rec, mask


def mae_vit_tiny_dec128d2b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

def mae_vit_small_dec128d2b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_vit_base_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_tiny = mae_vit_tiny_dec128d2b
mae_vit_small = mae_vit_small_dec128d2b
mae_vit_base = mae_vit_base_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large = mae_vit_large_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge = mae_vit_huge_dec512d8b  # decoder: 512 dim, 8 blocks
if __name__ == '__main__':
    import numpy as np

    model = mae_vit_tiny(img_size=32, patch_size=2, num_classes=10).to("cuda")

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1e6
    print('Trainable Parameters: %.3fM' % parameters)
