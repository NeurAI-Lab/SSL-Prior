from augmentations.helper import *
from torchvision import transforms


class Test_transform:
    """
    Transform defined in SimCLR
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, size, data_args):
        normalize = norm_mean_std(size)

        if data_args.name == 'STLTint':
            transform = [
                    transforms.ToPILImage(),
                    transforms.Resize(size=(size, size)),
                    transforms.ToTensor(),
                ]
        else:
            transform = [
                    transforms.Resize(size=(size, size)),
                    transforms.ToTensor(),
                ]
        if data_args.norm_aug:
            transform.append(normalize)
        self.transform = transforms.Compose(transform)

    def __call__(self, x):
        return self.transform(x)


