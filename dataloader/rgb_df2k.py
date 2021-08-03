from glob import glob

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DF2KRGBTrain(Dataset):
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
        lr_img = hr_img.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
        hr_img = (np.transpose(np.array(hr_img), (2, 0, 1)) / 255.) * 2 - 1
        lr_img = (np.transpose(np.array(lr_img), (2, 0, 1)) / 255.) * 2 - 1
        return torch.FloatTensor(np.array(lr_img)), torch.FloatTensor(np.array(hr_img))


class DF2KRGBValid(Dataset):
    def __init__(self, config):
        self.scale = config.scale
        self.lr_patch_size = config.lr_patch_size * self.scale
        self.hr_patch_size = self.lr_patch_size * self.scale
        self.valid_path = sorted(glob(config.valid_imgs))

    def __len__(self):
        return len(self.valid_path)

    def __getitem__(self, item):
        img_path = self.valid_path[item]
        hr_img = Image.open(img_path)
        hr_img = transforms.RandomCrop(self.hr_patch_size)(hr_img)
        lr_img = hr_img.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
        hr_img = (np.transpose(np.array(hr_img), (2, 0, 1)) / 255.) * 2 - 1
        lr_img = (np.transpose(np.array(lr_img), (2, 0, 1)) / 255.) * 2 - 1
        return torch.FloatTensor(lr_img), torch.FloatTensor(hr_img)


class DF2KLabelTrain(Dataset):
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
        lr_img = hr_img.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
        hr_img = np.transpose(np.array(hr_img), (2, 0, 1))
        lr_img = (np.transpose(np.array(lr_img), (2, 0, 1)) / 255.) * 2 - 1
        return torch.FloatTensor(np.array(lr_img)), torch.LongTensor(np.array(hr_img))


class DF2KLabelValid(Dataset):
    def __init__(self, config):
        self.scale = config.scale
        self.lr_patch_size = config.lr_patch_size
        self.hr_patch_size = self.lr_patch_size * self.scale
        self.valid_path = sorted(glob(config.valid_imgs))

    def __len__(self):
        return len(self.valid_path)

    def __getitem__(self, item):
        img_path = self.valid_path[item]
        hr_img = Image.open(img_path)
        hr_img = transforms.RandomCrop(self.hr_patch_size)(hr_img)
        lr_img = hr_img.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
        hr_img = np.transpose(np.array(hr_img), (2, 0, 1))
        lr_img = np.transpose(np.array(lr_img), (2, 0, 1))
        return torch.FloatTensor(np.array(lr_img)), torch.FloatTensor(np.array(hr_img))


if __name__ == '__main__':
    from config import Config as _config
    from torch.utils.data import DataLoader

    valid_dataset = DF2KRGBValid(_config)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=_config.batch_size,
                              shuffle=False)
    for i, data in enumerate(valid_loader):
        lr, hr = data
        print(lr.shape, hr.shape)
        print(torch.min(lr), torch.max(lr))
        print(torch.min(hr), torch.max(hr))
        break

    train_dataset = DF2KRGBTrain(_config)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=_config.batch_size,
                              shuffle=True,
                              num_workers=2)
    for i, data in enumerate(train_loader):
        lr, hr = data
        print(lr.shape, hr.shape)
        print(torch.min(lr), torch.max(lr))
        print(torch.min(hr), torch.max(hr))
        exit()
