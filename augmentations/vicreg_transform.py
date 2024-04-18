import numpy as np
from augmentations.helper import *
import torchvision
from torchvision import transforms
from PIL import ImageOps, ImageFilter
from util import registry

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

@registry.TRANSFORMS.register("vicreg_transform")
class VICRegTransform:
    """
    Transform defined in SimCLR
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, data_args):

        normalize = norm_mean_std(data_args)
        if data_args.transform == 'v1':
            transform = [
                transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ]
        else:
            transform = [
                    transforms.RandomResizedCrop(
                        size=data_args.img_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                            )
                        ],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(p=1.0),
                    Solarization(p=0.0),
                    transforms.ToTensor(),
                ]
        if data_args.norm_aug:
            transform.append(normalize)
        self.transform = transforms.Compose(transform)

        if data_args.transform == 'v1':
            transform_prime = [
                transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ]
        else:
            transform_prime = [
                    transforms.RandomResizedCrop(
                        size=data_args.img_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                            )
                        ],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(p=0.1),
                    Solarization(p=0.2),
                    transforms.ToTensor(),
                ]
        if data_args.norm_aug:
            transform_prime.append(normalize)
        self.transform_prime = transforms.Compose(transform_prime)

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2


