# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import timm.models.swin_transformer


class SwinTransformer(timm.models.swin_transformer.SwinTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, **kwargs):
        super(SwinTransformer, self).__init__(**kwargs)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def swin_tiny(**kwargs):
    model = SwinTransformer(
        patch_size=4, in_chans=3, global_pool='avg',
        embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), head_dim=None,
        mlp_ratio=4., qkv_bias=True,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def swin_small(**kwargs):
    model = SwinTransformer(
        patch_size=4, in_chans=3, global_pool='avg',
        embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), head_dim=None,
        mlp_ratio=4., qkv_bias=True,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


if __name__ == '__main__':
    import numpy as np

    img_size = 256
    img = torch.ones([2, 3, img_size, img_size]).to("cuda")

    model = swin_tiny(img_size=img_size, window_size=img_size // 16, num_classes=1000).to("cuda")
    # model = timm.models.swin_transformer.swin_tiny_patch4_window7_224().to("cuda")

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1e6
    print('Trainable Parameters: %.3fM' % parameters)

    predicted_class = model(img)

    print("Shape of predicted_class :", predicted_class.shape)  # [B, num_classes]
