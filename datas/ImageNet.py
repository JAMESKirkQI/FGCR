# coding=utf-8
# Copyright 2022 the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import os
import shutil

import PIL
import torch
from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision import datasets
from datas.classes import IMAGENET2012_CLASSES

_CITATION = """\
@article{imagenet15russakovsky,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = { {ImageNet Large Scale Visual Recognition Challenge} },
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
}
"""

_HOMEPAGE = "https://image-net.org/index.php"

_DESCRIPTION = """\
ILSVRC 2012, commonly known as 'ImageNet' is an image datasets organized according to the WordNet hierarchy. Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". There are more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). ImageNet aims to provide on average 1000 images to illustrate each synset. Images of each concept are quality-controlled and human-annotated. In its completion, ImageNet hopes to offer tens of millions of cleanly sorted images for most of the concepts in the WordNet hierarchy. ImageNet 2012 is the most commonly used subset of ImageNet. This datasets spans 1000 object classes and contains 1,281,167 training images, 50,000 validation images and 100,000 test images
"""


class ImageNet_1k(Dataset):
    def __init__(self, path=r'/to/your/path/ImageNet', train="train", transform=None):
        super().__init__()
        assert os.path.exists(os.path.join(path, 'train')) and os.path.exists(
            os.path.join(path, 'valid')), "please exam your Imagenet datasets"
        self.train = train
        self.train_list = os.listdir(os.path.join(path, 'train'))
        self.valid_list = os.listdir(os.path.join(path, 'valid'))
        self.imagenet1k = []
        if self.train == "train":
            for cls in self.train_list:
                self.imagenet1k.extend(glob.glob(os.path.join(path, "train", cls, "*.JPEG")))
        else:
            for cls in self.valid_list:
                self.imagenet1k.extend(glob.glob(os.path.join(path, "valid", cls, "*.JPEG")))
            lines = open(os.path.join(path, "valid_labels.txt")).readlines()
            self.valid_label = [int(line[:-1]) for line in lines]

        self.train_dict = IMAGENET2012_CLASSES
        self.train_index = list(self.train_dict.values())
        self.transform = transform

    def __getitem__(self, index):
        image = PIL.Image.open(self.imagenet1k[index])
        if image.size[0] != 3:
            image = image.convert('RGB')
        image = self.transform(image)
        label = self.train_dict[self.imagenet1k[index].split("/")[-2]]
        label = self.train_index.index(label)
        return image, label

    def __len__(self):
        return len(self.imagenet1k)

    def collate_fn(self, batch):
        images, label = zip(*batch)
        return images, label