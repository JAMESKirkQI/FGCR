import random
import math
from einops.einops import rearrange
import numpy as np
import torchvision.transforms as transforms


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio, regular=False):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.regular = regular

        if regular:
            assert mask_ratio == 0.75

            candidate_list = []
            while True:  # add more
                for j in range(4):
                    candidate = np.ones(4)
                    candidate[j] = 0
                    candidate_list.append(candidate)
                if len(candidate_list) * 4 >= self.num_patches * 2:
                    break
            self.mask_candidate = np.vstack(candidate_list)
            print('using regular, mask_candidate shape = ',
                  self.mask_candidate.shape)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}, regular {}".format(
            self.num_patches, self.num_mask, self.regular
        )
        return repr_str

    def __call__(self):
        if not self.regular:
            mask = np.hstack([
                np.zeros(self.num_patches - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
        else:
            mask = self.mask_candidate.copy()
            np.random.shuffle(mask)
            # symmetry_mask=mask
            mask = rearrange(mask[:self.num_patches // 4], '(h w) (p1 p2) -> (h p1) (w p2)',
                             h=self.height // 2, w=self.width // 2, p1=2, p2=2)
            mask = mask.flatten()

        return mask


class MultiRandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio, regular=False, r_views=4):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.regular = regular
        self.r_views = r_views
        self.mask_tpye_list = np.array([[2, 3, 4, 5], [3, 4, 5, 2], [4, 5, 2, 3], [5, 2, 3, 4]])
        if regular:
            assert mask_ratio == 0.75

            candidate_list = []
            while True:  # add more
                for mask_tpye in self.mask_tpye_list:
                    if len(candidate_list) * 4 >= self.num_patches:
                        break
                    candidate_list.append(mask_tpye)
                if len(candidate_list) * 4 >= self.num_patches:
                    break
            self.mask_candidate = np.vstack(candidate_list)
            print('using regular, mask_candidate shape = ',
                  self.mask_candidate.shape)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}, regular {}".format(
            self.num_patches, self.num_mask, self.regular
        )
        return repr_str

    def __call__(self):
        if not self.regular:
            mask = np.hstack([
                np.zeros(self.num_patches - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
        else:
            mask = self.mask_candidate.copy()
            np.random.shuffle(mask)
            mask_list = []
            for i in range(2, 2 + self.r_views):
                mask_mutli = mask.copy()
                mask_mutli[mask_mutli != i] = 1
                mask_mutli[mask_mutli == i] = 0
                mask_list.append(mask_mutli)
            if self.r_views > 0:
                mask_list = np.stack(mask_list)

                mask = rearrange(mask_list, 'r (h w) (p1 p2) -> r (h p1) (w p2)',
                                 h=self.height // 2, w=self.width // 2, p1=2, p2=2)
                mask = mask.reshape((self.r_views, -1))
            else:
                mask = np.ones(1).reshape((1, 1))
        return mask


class MaskTransform(object):
    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(
                0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if not hasattr(args, 'mask_regular'):
            args.mask_regular = False

        self.masked_position_generator = RandomMaskingGenerator(
            args.token_size, args.mask_ratio, args.mask_regular
        )

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr = "(MaskTransform,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class MultiMaskTransform(object):
    def __init__(self, transforms, args):
        self.transform = transforms

        self.masked_position_generator = MultiRandomMaskingGenerator(
            args.token_size, args.mask_ratio, args.mask_regular, args.r_views
        )

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr = "(MaskTransform,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr
