from augmentations.helper import *
from util import registry
from torchvision import transforms

@registry.TRANSFORMS.register("simclr_transform")
class SimCLRTransform:
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
            # if data_args.img_size == 224:  # ImageNet
            #     self.transform = transforms.Compose(
            #         [
            #             transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
            #             transforms.RandomHorizontalFlip(),
            #             get_color_distortion(s=1.0),
            #             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            #             transforms.ToTensor(),
            #             normalize
            #         ]
            #     )
            transform = [
                    transforms.RandomResizedCrop(size=(data_args.img_size, data_args.img_size)),
                    transforms.RandomHorizontalFlip(),
                    get_color_distortion(s=1.0),
                    transforms.ToTensor(),
                ]

        if data_args.norm_aug:
            transform.append(normalize)
        self.transform = transforms.Compose(transform)


    def __call__(self, x):
        return self.transform(x), self.transform(x)


