# Source: https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py

import numpy as np
import random
from torchvision import transforms as T
from torchvision.transforms import functional as F


class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ColorJitter(T.ColorJitter):
    def __call__(self, image, target):
        return (super().__call__(image), super().__call__(target))


class ToTensor(object):
    def __call__(self, image, target):
        return (F.to_tensor(image), F.to_tensor(target))
