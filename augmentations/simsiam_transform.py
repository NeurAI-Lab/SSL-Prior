from augmentations.helper import *
from torchvision import transforms
from util import registry

@registry.TRANSFORMS.register("simsiam_transform")
class SimSiamTransform:
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
        if data_args.transform == 'shv1':
            transform = [
                transform_sobel_edge(data_args, data_args.img_size * 2),
                transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ]
        elif data_args.transform == 'shv2':
            transform = [
                    transform_sobel_edge(data_args, data_args.img_size * 2),
                    transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
                    transforms.RandomHorizontalFlip(),
                    get_color_distortion(s=0.5),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                ]
        else:
            # if data_args.img_size == 224:  # ImageNet
            #     transform = [
            #             transforms.RandomResizedCrop(size=data_args.img_size),
            #             transforms.RandomHorizontalFlip(),
            #             get_color_distortion(s=0.5),
            #             transforms.RandomGrayscale(p=0.2),
            #             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            #             transforms.ToTensor(),
            #         ]
            #     if data_args.norm_aug:
            #         transform.append(normalize)
            #     self.transform = transforms.Compose(transform)
            transform = [
                    transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
                    transforms.RandomHorizontalFlip(),
                    get_color_distortion(s=0.5),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                ]

        if data_args.norm_aug:
            transform.append(normalize)
        self.transform = transforms.Compose(transform)

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


