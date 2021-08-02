from glob import glob

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


class DF2K(Dataset):
    def __init__(self, config):
        self.scale = config.scale
        self.lr_patch_size = config.lr_patch_size
        self.hr_patch_size = config.lr_patch_size * self.scale
        self.train_paths = glob(config.div2k)
        self.train_paths.extend(glob(config.flickr))
        self.train_paths = sorted(self.train_paths)

    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, item):
        img_path = self.train_paths[item]
        hr_img = Image.open(img_path)
        hr_img = transforms.RandomCrop(self.hr_patch_size)(hr_img)
        lr_img = transforms.Resize(self.lr_patch_size)(hr_img)
        hr_img = np.transpose(np.array(hr_img), (2, 0, 1))
        lr_img = np.transpose(np.array(lr_img), (2, 0, 1))
        return torch.FloatTensor(np.array(lr_img)), torch.FloatTensor(np.array(hr_img))


class DF2KLAB(Dataset):
    def __init__(self, config):
        self.scale = config.scale
        self.lr_patch_size = config.lr_patch_size
        self.hr_patch_size = config.lr_patch_size * self.scale
        self.train_paths = glob(config.div2k)
        self.train_paths.extend(glob(config.flickr))
        self.train_paths = sorted(self.train_paths)

    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, item):
        img_path = self.train_paths[item]
        hr_img = Image.open(img_path)
        hr_img = transforms.RandomCrop(self.hr_patch_size)(hr_img)
        lr_img = transforms.Resize(self.lr_patch_size)(hr_img)
        label = np.eye(256)[np.array(hr_img)]
        hr_img = rgb2lab(hr_img)
        lr_img = rgb2lab(lr_img)
        hr_img = np.transpose(np.array(hr_img), (2, 0, 1))
        lr_img = np.transpose(np.array(lr_img), (2, 0, 1))
        return torch.FloatTensor(np.array(lr_img)), \
               torch.FloatTensor(np.array(hr_img)),\
               torch.FloatTensor(np.array(label)),


if __name__ == '__main__':
    from config import Config as _config
    dataset = DF2KLAB(_config)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset=dataset,
                              batch_size=_config.batch_size,
                              shuffle=True,
                              num_workers=2)
    for i, data in enumerate(train_loader):
        lr, hr, label = data
        print(i, lr.shape, hr.shape, label.shape)
        exit()
