# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
import torch

from datas.DomainNet import DomainNet
from datas.Pmnist import Pmnist
from datas.chaoyang import CHAOYANG
from datas.ImageNet import ImageNet_1k
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import numpy as np

from util.mask_transform import MultiMaskTransform


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if "vit" not in args.model and is_train and not args.scratch:
        transform = MultiMaskTransform(transform, args)
    if args.data_path == 'c10':
        dataset = datasets.CIFAR10(
            root='/to/your/path/CIFAR10', train=is_train, download=True, transform=transform
        )
    elif args.data_path == 'c100':
        dataset = datasets.CIFAR100(
            root='/to/your/path/CIFAR100', train=is_train, download=False, transform=transform
        )
    elif args.data_path == 'svhn':
        split = 'train' if is_train else 'test'
        dataset = datasets.SVHN(root='/to/your/path/svhn', split=split, download=True, transform=transform)
    elif args.data_path == 'imagenet':
        split = 'train' if is_train else 'val'
        dataset = ImageNet_1k(path='/to/your/path/ImageNet', train=split, transform=transform)
    elif args.data_path == 'sketch':
        split = 'train' if is_train else 'val'
        dataset = DomainNet(path='/to/your/path/DomainNet', dataset="sketch", train=split,
                            transform=transform)
    elif args.data_path == 'clipart':
        split = 'train' if is_train else 'val'
        dataset = DomainNet(path='/to/your/path/DomainNet', dataset="clipart", train=split,
                            transform=transform)
    elif args.data_path == 'infograph':
        split = 'train' if is_train else 'val'
        dataset = DomainNet(path='/to/your/path/DomainNet', dataset="infograph", train=split,
                            transform=transform)
    elif args.data_path == 'painting':
        split = 'train' if is_train else 'val'
        dataset = DomainNet(path='/to/your/path/DomainNet', dataset="painting", train=split,
                            transform=transform)
    elif args.data_path == 'quickdraw':
        split = 'train' if is_train else 'val'
        dataset = DomainNet(path='/to/your/path/DomainNet', dataset="quickdraw", train=split,
                            transform=transform)
    elif args.data_path == 'real':
        split = 'train' if is_train else 'val'
        dataset = DomainNet(path='/to/your/path/DomainNet', dataset="real", train=split,
                            transform=transform)
    elif args.data_path == 'chaoyang':
        dataset = CHAOYANG(path='/to/your/path/chaoyang', train=is_train, transform=transform)
    elif args.data_path == 'pmnist':
        split = 'train' if is_train else 'val'
        dataset = Pmnist(path='/to/your/path/pmnist', train=split, transform=transform)
    elif args.data_path == 'flowers102':
        split = 'train' if is_train else 'test'
        root = os.path.join('/to/your/path/flowers102', split)
        dataset = datasets.ImageFolder(root, transform=transform)
    else:
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)

    if is_train and args.subset_size != 1:
        np.random.seed(127)

        dsize = len(dataset)
        idxs = np.random.choice(dsize, int(dsize * args.subset_size), replace=False)

        dataset = torch.utils.data.Subset(dataset, idxs)
        print('Subset dataset size: ', len(dataset))

    print(f'{dataset} ({len(dataset)})')

    return dataset


def build_transform(is_train, args):
    if args.data_path == 'c10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    if 'perturb_perspective' in vars(args) and args.perturb_perspective:
        print('[Log] Perturbing perspective of images in dataset')
        t.append(transforms.RandomPerspective(distortion_scale=0.5, p=1.0))
    else:
        print('[Log] Not perturbing perspective of images in dataset')

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
