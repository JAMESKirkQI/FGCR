import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Pmnist(Dataset):
    def __init__(self, path=r'/to/your/path/pmnist', train="train", transform=None):
        assert os.path.join(path, "pneumoniamnist.npz"), "get in https://github.com/MedMNIST/MedMNIST"
        npz_file = np.load(os.path.join(path, "pneumoniamnist.npz"))
        self.split = train
        self.transform = transform
        self.target_transform = transform

        if self.split == "train":
            self.imgs = npz_file[f"train_images"]
            self.imgs_val = npz_file[f"val_images"]
            self.labels = npz_file[f"train_labels"]
            self.labels_val = npz_file[f"val_labels"]
            self.imgs = np.concatenate([self.imgs, self.imgs_val], axis=0)
            self.labels = np.concatenate([self.labels, self.labels_val], axis=0)
        else:
            self.imgs = npz_file[f"test_images"]
            self.labels = npz_file[f"test_labels"]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index][0].astype(int)
        img = Image.fromarray(img)
        if img.size[0] != 3:
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


if __name__ == '__main__':
    pmnist = Pmnist()
    print(len(pmnist))
    print(pmnist[0])
    pmnist = Pmnist(train="test")
    print(len(pmnist))
