from PIL import ImageFilter
import random
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from PIL import ImageOps, ImageFilter
# from imagecorruptions import get_corruption_names, corrupt
#
# class ImageCorruptions:
#     def __init__(self, args):
#         self.severity = args.corrupt_severity
#         self.corruption_name = args.corrupt_name
#
#     def __call__(self, image, labels=None):
#
#         image = np.array(image)
#         cor_image = corrupt(image, corruption_name=self.corruption_name,
#                         severity=self.severity)
#
#         return Image.fromarray(cor_image)

def norm_mean_std(size):
    if size == 32:  # CIFAR10, CIFAR100
        normalize = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    elif size == 64:  # Tiny-ImageNet
        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif size == 96:  # STL10
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:  # ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize


class GaussianBlur(object):
    """Gaussian blur augmentation """

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class GaussianBlur_vic(object):
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

def get_color_distortion(s=1.0):
    """
    Color jitter from SimCLR paper
    @param s: is the strength of color distortion.
    """

    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

class transform_sobel_edge(object):
    def __init__(self, args, upsample_size=0):
        self.gauss_ksize = args.sobel_gauss_ksize
        self.sobel_ksize = args.sobel_ksize
        self.upsample = args.sobel_upsample
        self.upsample_size = upsample_size

    def __call__(self, img, boxes=None, labels=None):

        if self.upsample:
            curr_size = img.size[0]
            resize_up = transforms.Resize(max(curr_size, self.upsample_size), 3)
            resize_down = transforms.Resize(curr_size, 3)
            rgb = np.array(resize_up(img))
        else:
            rgb = np.array(img)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        rgb = cv2.GaussianBlur(rgb, (self.gauss_ksize, self.gauss_ksize), self.gauss_ksize)
        sobelx = cv2.Sobel(rgb, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        imgx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.Sobel(rgb, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        imgy = cv2.convertScaleAbs(sobely)
        tot = np.sqrt(np.square(sobelx) + np.square(sobely))
        imgtot = cv2.convertScaleAbs(tot)
        sobel_img = Image.fromarray(cv2.cvtColor(imgtot, cv2.COLOR_GRAY2BGR))
        sobel_img = resize_down(sobel_img) if self.upsample else sobel_img

        return sobel_img

class transform_canny_edge(object):

    def __init__(self, args):
        self.gauss_ksize = args.sobel_gauss_ksize

    def __call__(self, img):
        rgb = np.array(img)
        image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (self.gauss_ksize, self.gauss_ksize), self.gauss_ksize)
        out = cv2.Canny(image, 50, 100)
        # out = np.stack([edges, edges, edges], axis=-1)
        canny_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_GRAY2BGR))
        # to_pil = transforms.ToPILImage()
        # out = to_pil(out)
        return canny_img

class transform_prewitt_edge(object):
    def __init__(self, args, upsample_size=0):
        self.gauss_ksize = args.sobel_gauss_ksize
        self.sobel_ksize = args.sobel_ksize
        self.upsample = args.sobel_upsample
        self.upsample_size = upsample_size

    def __call__(self, img, boxes=None, labels=None):

        if self.upsample:
            curr_size = img.size[0]
            resize_up = transforms.Resize(max(curr_size, self.upsample_size), 3)
            resize_down = transforms.Resize(curr_size, 3)
            rgb = np.array(resize_up(img))
        else:
            rgb = np.array(img)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        rgb = cv2.GaussianBlur(rgb, (self.gauss_ksize, self.gauss_ksize), self.gauss_ksize)

        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        img_prewittx = cv2.filter2D(rgb, -1, kernelx)
        img_prewitty = cv2.filter2D(rgb, -1, kernely)

        tot = img_prewittx + img_prewitty
        imgtot = cv2.convertScaleAbs(tot)
        prewitt_img = Image.fromarray(cv2.cvtColor(imgtot, cv2.COLOR_GRAY2BGR))

        prewitt_img = resize_down(prewitt_img) if self.upsample else prewitt_img

        return prewitt_img