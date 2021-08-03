from glob import glob

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DF2KLabTrain(Dataset):
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
        hr_rgb = Image.open(img_path)
        hr_rgb = transforms.RandomCrop(self.hr_patch_size)(hr_rgb)
        lr_img = hr_rgb.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
        hr_lab = rgb2lab(hr_rgb)

        hr_rgb = np.transpose(np.array(hr_rgb), (2, 0, 1))
        lr_img = (np.transpose(np.array(lr_img), (2, 0, 1)) / 255.) * 2 - 1

        return torch.FloatTensor(np.array(lr_img)), \
               torch.LongTensor(np.array(hr_rgb)),\
               torch.FloatTensor(np.array(hr_lab)),


class DF2KLabValid(Dataset):
    def __init__(self, config):
        self.scale = config.scale
        self.lr_patch_size = config.lr_patch_size
        self.hr_patch_size = config.lr_patch_size * self.scale
        self.valid_path = sorted(glob(config.valid_imgs))

    def __len__(self):
        return len(self.valid_path)

    def __getitem__(self, item):
        img_path = self.valid_path[item]
        hr_rgb = Image.open(img_path)
        hr_rgb = transforms.RandomCrop(self.hr_patch_size)(hr_rgb)
        lr_img = hr_rgb.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
        hr_rgb = np.transpose(np.array(hr_rgb), (2, 0, 1))
        lr_img = (np.transpose(np.array(lr_img), (2, 0, 1)) / 255.) * 2 - 1
        return torch.FloatTensor(np.array(lr_img)), \
               torch.FloatTensor(np.array(hr_rgb))


class DF2KLabOnlyTrain(Dataset):
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
        hr_rgb = Image.open(img_path)
        hr_rgb = transforms.RandomCrop(self.hr_patch_size)(hr_rgb)
        lr_rgb = hr_rgb.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
        hr_lab = rgb2lab(hr_rgb)
        lr_lab = rgb2lab(lr_rgb)
        hr_lab = np.transpose(np.array(hr_lab), (2, 0, 1))
        lr_lab = np.transpose(np.array(lr_lab), (2, 0, 1))
        return torch.FloatTensor(np.array(lr_lab)), \
               torch.FloatTensor(np.array(hr_lab)),


class DF2KLabOnlyValid(Dataset):
    def __init__(self, config):
        self.scale = config.scale
        self.lr_patch_size = config.lr_patch_size
        self.hr_patch_size = config.lr_patch_size * self.scale
        self.valid_path = sorted(glob(config.valid_imgs))

    def __len__(self):
        return len(self.valid_path)

    def __getitem__(self, item):
        img_path = self.valid_path[item]
        hr_rgb = Image.open(img_path)
        hr_rgb = transforms.RandomCrop(self.hr_patch_size)(hr_rgb)
        lr_rgb = hr_rgb.resize((self.lr_patch_size, self.lr_patch_size), Image.BICUBIC)
        lr_lab = rgb2lab(lr_rgb)
        lr_lab = np.transpose(np.array(lr_lab), (2, 0, 1))
        return torch.FloatTensor(np.array(lr_lab)), \
               torch.FloatTensor(np.array(hr_rgb))


if __name__ == '__main__':
    from config import Config as _config
    from torch.utils.data import DataLoader
    _config.div2k = f'../dataset/DIV2K_train_HR/*.png'
    _config.flickr = f'../dataset/Flickr2K/*.png'
    train_dataset = DF2KLabTrain(_config)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=_config.batch_size,
                              shuffle=True,
                              num_workers=2)
    for i, data in enumerate(train_loader):
        lr, label, lab = data
        print(i, lr.shape, label.shape, lab.shape, "?")
        print(torch.min(lr), torch.max(lr))
        print(torch.min(label), torch.max(label))
        print(torch.min(lab[0][..., 0]), torch.max(lab[0][..., 0]))
        print(torch.min(lab[0][..., 1]), torch.max(lab[0][..., 1]))
        print(torch.min(lab[0][..., 2]), torch.max(lab[0][..., 2]))
        exit()
