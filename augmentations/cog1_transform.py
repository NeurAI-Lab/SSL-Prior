from augmentations.helper import *
from torchvision import transforms
from util import registry

@registry.TRANSFORMS.register("cog1_transform")
class cog1Transform:
    """
    Transform defined in SimCLR
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, data_args):
        normalize = norm_mean_std(data_args)
        transform = [
            transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]
        if data_args.norm_aug:
            transform.append(normalize)
        self.transform = transforms.Compose(transform)

        transform_prime = [
            transform_sobel_edge(data_args, data_args.img_size*2),
            transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]
        if data_args.norm_aug:
            transform_prime.append(normalize)
        self.transform_prime = transforms.Compose(transform_prime)

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return [x1, x2]


