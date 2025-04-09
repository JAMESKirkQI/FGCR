import glob
import os

import PIL
from torch.utils.data import Dataset


class DomainNet(Dataset):
    def __init__(self, path=r'/to/your/path/DomainNet', dataset="clipart", train="train", transform=None):
        super().__init__()
        assert os.path.exists(os.path.join(path, dataset)) and os.path.exists(
            os.path.join(path, dataset + "_train.txt")) and os.path.exists(
            os.path.join(path, dataset + "_test.txt")), "please exam your path"
        self.train = train
        self.path = path
        if self.train == "train":
            with open(os.path.join(path, dataset + "_train.txt"), 'r') as file:
                self.DomainNet_list = file.readlines()
        else:
            with open(os.path.join(path, dataset + "_test.txt"), 'r') as file:
                self.DomainNet_list = file.readlines()
        self.transform = transform

    def __getitem__(self, index):
        image_path, label = self.DomainNet_list[index].split(" ")
        image_path = os.path.join(self.path, image_path)
        image = PIL.Image.open(image_path)
        if image.size[0] != 3:
            image = image.convert('RGB')
        image = self.transform(image)

        label = int(label)
        return image, label

    def __len__(self):
        return len(self.DomainNet_list)

    def collate_fn(self, batch):
        images, label = zip(*batch)
        return images, label
