import numpy as np
from scipy import linalg
from skimage.io import imread
from skimage.color import rgb2lab, rgb2xyz, xyz2lab, lab2rgb, lab2xyz, xyz2rgb

import torch
from utils import tv_rgb2xyz, tv_xyz2lab


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rgb_img = imread('tmp.jpg')
    xyz_img = rgb2xyz(rgb_img)
    lab_img = xyz2lab(xyz_img)
    print(np.sum(lab_img), lab_img.shape)
    # lab_img = rgb2lab(rgb_img)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    rgb_tensor = torch.FloatTensor(np.transpose(rgb_img, (2, 0, 1))).to(device)
    xyz_tensor = tv_rgb2xyz(rgb_tensor, device)
    lab_tensor = tv_xyz2lab(xyz_tensor, device)
    print(torch.sum(lab_tensor), lab_tensor.shape)
