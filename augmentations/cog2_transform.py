from augmentations.helper import *
from torchvision import transforms
from util import registry

@registry.TRANSFORMS.register("cog2_transform")
class cog2Transform:
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
            transform_prime = [
                transform_sobel_edge(data_args, data_args.img_size*2),
                transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ]
        elif data_args.transform == 'v2':
            transform = [
                transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
                transforms.RandomHorizontalFlip(),
                get_color_distortion(s=1.0),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                ]
            transform_prime = [
                transform_sobel_edge(data_args, data_args.img_size*2),
                transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ]
        elif data_args.transform == 'v3':
            transform = [
                transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
                transforms.RandomHorizontalFlip(),
                get_color_distortion(s=1.0),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                ]
            transform_prime = [
                transform_sobel_edge(data_args, data_args.img_size*2),
                transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
                transforms.RandomHorizontalFlip(),
                get_color_distortion(s=1.0),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                ]

        if data_args.norm_aug:
            transform.append(normalize)
        if data_args.norm_aug:
            transform_prime.append(normalize)
        self.transform = transforms.Compose(transform)
        self.transform_prime = transforms.Compose(transform_prime)

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2


