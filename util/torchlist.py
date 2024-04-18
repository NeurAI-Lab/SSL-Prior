"""
@author: Fahad Sarfraz
"""
import torch.utils.data as data
from PIL import Image
import os
import os.path


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist, sep):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split(sep)
            imlist.append( (impath, int(imlabel)) )

    return imlist

def folder_reader(data_dir):
    all_img_files = []
    all_labels = []

    class_names = os.walk(data_dir).__next__()[1]
    for index, class_name in enumerate(class_names):
        label = index
        img_dir = os.path.join(data_dir, class_name)
        img_files = os.walk(img_dir).__next__()[2]

        for img_file in img_files:
            img_file = os.path.join(img_dir, img_file)
            img = Image.open(img_file)
            if img is not None:
                all_img_files.append(img_file)
                all_labels.append(int(label))

    return all_img_files, all_labels

def subset_folder_reader(data_dir, flist):
    all_img_files = []
    all_labels = []

    with open(flist, 'r') as rf:
        for line in rf.readlines():
            imfolder, imlabel = line.strip().split(' ')
            class_name = imfolder
            label = imlabel
            img_dir = os.path.join(data_dir, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(int(label))

    return all_img_files, all_labels

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist=None, folderlist=None, subset_folderlist=None, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader, sep=' '):
        self.root = root
        self.imlist = []
        if flist:
            self.imlist = flist_reader(flist,sep)
            #self.imlist = style_flist_reader(root, flist,sep)

        elif subset_folderlist:
            self.images, self.labels = subset_folder_reader(folderlist, subset_folderlist)
        else:
            self.images, self.labels = folder_reader(folderlist)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        if self.imlist:
            impath, target = self.imlist[index]
            img = self.loader(os.path.join(self.root ,impath))
        else:
            img = self.loader(self.images[index])
            target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist) if self.imlist else len(self.images)
