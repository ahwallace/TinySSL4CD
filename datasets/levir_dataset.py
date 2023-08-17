import os
import random
from PIL import Image
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from torchvision import transforms
import torchvision.transforms.functional as TF

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class LevirDataset(Dataset):
    """ LEVIR-CD dataset with pixel-level labels.
        ├─A
        ├─B
        ├─label
        └─list

        Slight modification of: https://github.com/AndreaCodegoni/Tiny_model_4_CD
    """

    def __init__(
            self,
            root_dir: str,
            split: str = 'train',
            pretrain=False
        ):
        super().__init__()

        self.root_dir = root_dir
        self.A = os.path.join(root_dir, "A")
        self.B = os.path.join(root_dir, "B")
        self.label = os.path.join(root_dir, "label")

        self.split = split
        self.pretrain = pretrain

        self.list_images = self.read_images_list()

        # Transforms
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=[5, 9], sigma=[0.1, 5])
        self.normalize = transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
                                              std=torch.tensor(IMAGENET_DEFAULT_STD))
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        # Current image set name:
        imgname = self.list_images[idx].strip('\n')

        # Loading the images:
        x_ref = Image.open(os.path.join(self.A, imgname))
        x_post = Image.open(os.path.join(self.B, imgname))
        x_mask = Image.open(os.path.join(self.label, imgname))

        # Data augmentation in case of training:
        if self.split == 'train':
            x_ref, x_post, x_mask = self.transform(x_ref, x_post, x_mask)

        x_ref, x_post, x_mask = [self.to_tensor(x) for x in (x_ref, x_post, x_mask)]
        x_ref, x_post = [self.normalize(x) for x in (x_ref, x_post)]

        return (x_ref, x_post), x_mask

    def __len__(self):
        return len(self.list_images)

    def read_images_list(self) -> List[str]:
        images_list_file = os.path.join(self.root_dir, 'list', self.split + ".txt")
        with open(images_list_file, "r") as f:
            return f.readlines()

    def transform(self, x_ref, x_post, x_mask
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if random.random() > 0.5:
            x_ref, x_post, x_mask = [TF.hflip(x) for x in (x_ref, x_post, x_mask)]

        if random.random() > 0.5:
            x_ref, x_post, x_mask = [TF.vflip(x) for x in (x_ref, x_post, x_mask)]
        
        if not self.pretrain:
            if random.random() > 0.5:
                angle = transforms.RandomRotation.get_params(degrees=[-5, 5])
                x_ref, x_post, x_mask = [TF.rotate(x, angle=angle) for x in (x_ref, x_post, x_mask)]

            # Apply colour transforms to x_ref and x_post seperately
            if random.random() > 0.5:
                x_ref, x_post = [self.color_jitter(x) for x in (x_ref, x_post)]

            if random.random() > 0.5:
                x_ref, x_post = [self.gaussian_blur(x) for x in (x_ref, x_post)]

        return x_ref, x_post, x_mask