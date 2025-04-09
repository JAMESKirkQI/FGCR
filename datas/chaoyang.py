import torch.utils.data as data
from PIL import Image
import os
import json


class CHAOYANG(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        imgs = []
        labels = []
        if train:
            json_path = os.path.join(path, "train.json")
        else:
            json_path = os.path.join(path, "test.json")
        with open(json_path, 'r') as f:
            load_list = json.load(f)
            for i in range(len(load_list)):
                img_path = os.path.join(path, load_list[i]["name"])
                imgs.append(img_path)
                labels.append(load_list[i]["label"])
        self.transform = transform
        self.train = train  # training set or test set
        self.dataset = 'chaoyang'
        self.nb_classes = 4

        self.data, self.labels = imgs, labels

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)
