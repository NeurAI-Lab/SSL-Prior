import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
from copy import deepcopy
from torchvision import datasets
from torch.utils.data import DataLoader,TensorDataset
from util.torchlist import ImageFilelist

SAVE_IMG = False
VALID_SPURIOUS = [
    'TINT',  # apply a fixed class-wise tinting (meant to not affect shape)
]

def celeb_indicies(split, ds, attr_names_map, unlabel_skew=True):
    male_mask = ds.attr[:, attr_names_map['Male']] == 1
    female_mask = ds.attr[:, attr_names_map['Male']] == 0
    blond_mask = ds.attr[:, attr_names_map['Blond_Hair']] == 1
    not_blond_mask = ds.attr[:, attr_names_map['Blond_Hair']] == 0

    indices = torch.arange(len(ds))

    if split == 'train':
        male_blond = indices[torch.logical_and(male_mask, blond_mask)]
        male_not_blond = indices[torch.logical_and(male_mask, not_blond_mask)]
        female_blond = indices[torch.logical_and(female_mask, blond_mask)]
        female_not_blond = indices[torch.logical_and(female_mask, not_blond_mask)]
        p_male_blond = len(male_blond) / float(len(indices))
        p_male_not_blond = len(male_not_blond) / float(len(indices))
        p_female_blond = len(female_blond) / float(len(indices))
        p_female_not_blond = len(female_not_blond) / float(len(indices))

        # training set must have 500 male_not_blond and 500 female_blond
        train_N = 1000
        training_male_not_blond = male_not_blond[:train_N]
        training_female_blond = female_blond[:train_N]

        unlabeled_male_not_blond = male_not_blond[train_N:]
        unlabeled_female_blond = female_blond[train_N:]
        unlabeled_male_blond = male_blond
        unlabeled_female_not_blond = female_not_blond

        if unlabel_skew:
            # take 1000 from each category
            unlabeled_N = 1000
            unlabeled_male_not_blond = unlabeled_male_not_blond[:unlabeled_N]
            unlabeled_female_blond = unlabeled_female_blond[:unlabeled_N]
            unlabeled_male_blond = unlabeled_male_blond[:unlabeled_N]
            unlabeled_female_not_blond = unlabeled_female_not_blond[:unlabeled_N]
        else:
            total_N = 4000
            extra = total_N - int(p_male_not_blond * total_N) - int(p_female_blond * total_N) - int(
                p_male_blond * total_N) - int(p_female_not_blond * total_N)
            unlabeled_male_not_blond = unlabeled_male_not_blond[:int(p_male_not_blond * total_N)]
            unlabeled_female_blond = unlabeled_female_blond[:int(p_female_blond * total_N)]
            unlabeled_male_blond = unlabeled_male_blond[:int(p_male_blond * total_N)]
            unlabeled_female_not_blond = unlabeled_female_not_blond[:(int(p_female_not_blond * total_N) + extra)]

        train_indices = np.concatenate([training_male_not_blond, training_female_blond])
        unlabelled_indices = np.concatenate(
            [unlabeled_male_not_blond, unlabeled_female_blond, unlabeled_male_blond, unlabeled_female_not_blond])
        for index in unlabelled_indices:
            assert index not in train_indices

        if split == 'train':
            indices = train_indices
        else:
            indices = unlabelled_indices

    imgs = []
    imgs_fp = []
    ys = []
    metas = []
    is_blonde= []
    for i in indices:

        img, attr = ds[i]
        imgs.append(img)
        ys.append(attr[attr_names_map['Male']])
        is_blonde.append(blond_mask[i])
        if male_mask[i]:
            agree = False if blond_mask[i] else True
        else:
            agree = True if blond_mask[i] else False
        metas.append({'agrees': agree, 'blond': blond_mask[i], 'male': male_mask[i]})

    print(split, len(indices))
    if split == 'test':
        return TensorDataset(torch.stack(imgs), torch.tensor(ys), torch.tensor(is_blonde))
    else:
        return TensorDataset(torch.stack(imgs), torch.tensor(ys))

def add_spurious(ds, mode):
    assert mode in VALID_SPURIOUS

    loader = DataLoader(ds, batch_size=32, num_workers=1,
                        pin_memory=False, shuffle=False)

    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    xs = torch.cat(xs)
    ys = torch.cat(ys)

    colors = torch.tensor([(2, 1, 0), (1, 2, 0), (1, 1, 0),
                           (0, 2, 1), (0, 1, 2), (0, 1, 1),
                           (1, 0, 2), (2, 0, 1), (1, 0, 1),
                           (1, 1, 1)])

    colors = colors / torch.sum(colors + 0.0, dim=1, keepdim=True)

    xs_tint = (xs + colors[ys].unsqueeze(-1).unsqueeze(-1) / 3).clamp(0, 1)

    return TensorDataset(xs_tint, ys)

class stl_tint(torch.utils.data.Dataset):
    def __init__(self, dataset, split, transform=None):
        self.dataset = dataset if split =='test' else add_spurious(dataset, 'TINT')
        self.dataset_orig = dataset

        self.transform = transform

    def __getitem__(self, index):

        img, target = self.dataset[index]

        if SAVE_IMG:
            import torchvision.transforms as transforms
            img_orig, _ = self.dataset_orig[index]
            plot(transforms.ToPILImage()(img_orig), 'stl')
            plot(transforms.ToPILImage()(img), 'stl_tint')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.dataset)

class STLTint():
    NUM_CLASSES = 10
    MEAN = [0.4192, 0.4124, 0.3804]
    STD = [0.2714, 0.2679, 0.2771]
    SIZE = 96
    SOBEL_UPSAMPLE_SIZE = 196

    def __init__(self):
        pass

    def get_dataset(self, data_dir, split, download=True, transform=None):

        if split == 'val_train':
            dataset = torchvision.datasets.STL10(root=data_dir, split='train', folds=None, download=True,
                                   transform=transforms.ToTensor())
        elif split == 'val_test':
            dataset = torchvision.datasets.STL10(root=data_dir, split='test', folds=None, download=True,
                                   transform=transforms.ToTensor())
        else:
            dataset = torchvision.datasets.STL10(root=data_dir, split=split, folds=None, download=True,
                                                 transform=transforms.ToTensor())
        if split == 'val_train':
            ds = stl_tint(dataset, split='train', transform=transform)
        elif split == 'val_test':
            ds = stl_tint(dataset, split='test', transform=transform)

        return ds

class CIFAR10ImbalancedNoisy(datasets.CIFAR10):
    """CIFAR100 dataset, with support for Imbalanced and randomly corrupt labels.

    Params
    ------
    corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
    num_classes: int
    Default 10. The number of classes in the dataset.
    """
    def __init__(self, corrupt=0.0, gamma=-1, n_min=25, n_max=500,
                 num_classes=10, perc=1.0, **kwargs):
        super(CIFAR10ImbalancedNoisy, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.perc = perc
        self.gamma = gamma
        self.corrupt = corrupt
        self.n_min = n_min
        self.n_max = n_max
        self.true_labels = deepcopy(self.targets)

        if perc < 1.0:
            print('*' * 30)
            print('Creating a Subset of Dataset')
            self.get_subset()
            (unique, counts) = np.unique(self.targets, return_counts=True)
            frequencies = np.asarray((unique, counts)).T
            print(frequencies)

        if gamma > 0:
            print('*' * 30)
            print('Creating Imbalanced Dataset')
            self.imbalanced_dataset()
            self.true_labels = deepcopy(self.targets)

        if corrupt > 0:
            print('*' * 30)
            print('Applying Label Corruption')
            self.corrupt_labels(corrupt)

    def get_subset(self):
        np.random.seed(12345)

        lst_data = []
        lst_targets = []
        targets = np.array(self.targets)
        for class_idx in range(self.num_classes):
            class_indices = np.where(targets == class_idx)[0]
            num_samples = int(self.perc * len(class_indices))
            sel_class_indices = class_indices[:num_samples]
            lst_data.append(self.data[sel_class_indices])
            lst_targets.append(targets[sel_class_indices])

        self.data = np.concatenate(lst_data)
        self.targets = np.concatenate(lst_targets)

        assert len(self.targets) == len(self.data)


    def imbalanced_dataset(self):
        np.random.seed(12345)
        X = np.array([[1, -self.n_max], [1, -self.n_min]])
        Y = np.array([self.n_max, self.n_min * self.num_classes ** (self.gamma)])

        a, b = np.linalg.solve(X, Y)

        classes = list(range(1, self.num_classes + 1))

        imbal_class_counts = []
        for c in classes:
          num_c = int(np.round(a / (b + (c) ** (self.gamma))))
          print(c, num_c)
          imbal_class_counts.append(num_c)

        print(imbal_class_counts)
        targets = np.array(self.targets)

        # Get class indices
        class_indices = [np.where(targets == i)[0] for i in range(self.num_classes)]

        # Get imbalanced number of instances
        imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
        imbal_class_indices = np.hstack(imbal_class_indices)

        np.random.shuffle(imbal_class_indices)

        # Set target and data to dataset
        self.targets = targets[imbal_class_indices]
        self.data = self.data[imbal_class_indices]

        assert len(self.targets) == len(self.data)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.num_classes, mask.sum())
        labels[mask] = rnd_labels

        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]
        self.targets = labels

class CelebA:
    NUM_CLASSES = 2
    CLASS_NAMES = ['female', 'male']
    MEAN = [0, 0, 0]
    STD = [1, 1, 1, ]
    SIZE = 96
    SOBEL_UPSAMPLE_SIZE = 196

    def __init__(self):
        pass

    def get_dataset(self, data_dir, split, download=True, transform=None, unlabel_skew=True):
        assert split in ['train', 'val', 'test', 'unlabeled']

        if split == 'test':
            ds = torchvision.datasets.CelebA(root=data_dir, split='test', transform=transform)
        # elif split == 'val':
        #     ds = torchvision.datasets.CelebA(root=self.data_path, split='valid', transform=trans)
        else:
            ds = torchvision.datasets.CelebA(root=data_dir, split='train', transform=transform)
        attr_names = ds.attr_names
        attr_names_map = {a: i for i, a in enumerate(attr_names)}
        ds = celeb_indicies(split, ds, attr_names_map, unlabel_skew)

        return ds

class Imagenet_R():
    NUM_CLASSES = 1000
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    SOBEL_UPSAMPLE_SIZE = 64

    def __init__(self):
        pass

    def get_dataset(self, data_dir, split, download=True, transform=None):
        ds = ImageFilelist(root=data_dir, flist=os.path.join(data_dir, "annotations.txt"),
                               transform=transform)

        return ds

class Imagenet_Blurry():
    NUM_CLASSES = 1000
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    SOBEL_UPSAMPLE_SIZE = 64

    def __init__(self):
        pass

    def get_dataset(self, data_dir, split, download=True, transform=None):
        ds = ImageFilelist(root=data_dir, flist=os.path.join(data_dir, "annotations.txt"),
                           transform=transform)

        return ds

class Imagenet_A():
    NUM_CLASSES = 1000
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    SOBEL_UPSAMPLE_SIZE = 64

    def __init__(self):
        pass

    def get_dataset(self, data_dir, split, download=True, transform=None):
        ds = ImageFilelist(root=data_dir, flist=os.path.join(data_dir, "annotations.txt"),
                           transform=transform, sep=',')

        return ds

class Imagenet_C():
    NUM_CLASSES = 1000
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 64
    SOBEL_UPSAMPLE_SIZE = 64

    def __init__(self):
        pass

    def get_dataset(self, data_dir, split, download=True, transform=None):
        ds = ImageFilelist(root=data_dir, flist=os.path.join(data_dir, "annotations.txt"),
                           transform=transform)

        return ds

class Imagenet200():
    NUM_CLASSES = 200
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 256

    def get_dataset(self, data_dir, split, download=True, transform=None):
        assert split in ['train', 'test']

        if split == 'test':
            ds = ImageFilelist(root=self.data_path, folderlist=os.path.join(self.data_path, "val"), subset_folderlist=os.path.join(self.data_path, "imgnet200.txt"),
                               transform=transform)
        else:
            ds = ImageFilelist(root=self.data_path, folderlist=os.path.join(self.data_path, "train"), subset_folderlist=os.path.join(self.data_path, "imgnet200.txt"),
                               transform=transform)

        return ds


class Imagenet100():
    NUM_CLASSES = 100
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 256

    def get_dataset(self, data_dir, split, download=True, transform=None):
        assert split in ['train', 'test']

        if split == 'test':
            ds = ImageFilelist(root=data_dir, folderlist=os.path.join(data_dir, "val"), transform=transform)
        else:
            ds = ImageFilelist(root=data_dir, folderlist=os.path.join(data_dir, "train"), transform=transform)

        return ds

class Imagenette():
    NUM_CLASSES = 10
    MEAN = [0.4802, 0.4481, 0.3975]
    STD = [0.2302, 0.2265, 0.2262]
    SIZE = 256

    def get_dataset(self, data_dir, split, download=True, transform=None):
        assert split in ['train', 'test']

        if split == 'test':
            ds = ImageFilelist(root=data_dir, folderlist=os.path.join(data_dir, "val"), transform=transform)
        else:
            ds = ImageFilelist(root=data_dir, folderlist=os.path.join(data_dir, "train"), transform=transform)

        return ds