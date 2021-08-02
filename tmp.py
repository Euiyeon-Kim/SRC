import cv2
import numpy as np
from scipy import linalg
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb, lab2xyz

import torch

lab_ref_white = (0.95047, 1., 1.08883)
xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])
rgb_from_xyz = linalg.inv(xyz_from_rgb)

if __name__ == '__main__':
    path = 'test.png'
    rgb_numpy = imread(path)
    lab_numpy = rgb2lab(rgb_numpy)
    l_numpy, a_numpy, b_numpy = lab_numpy[..., 0], lab_numpy[..., 1], lab_numpy[..., 2]

    rgb_tensor = torch.FloatTensor(rgb_numpy)
    l_tensor = torch.FloatTensor(l_numpy)
    a_tensor = torch.FloatTensor(a_numpy)
    b_tensor = torch.FloatTensor(b_numpy)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    y = (l_numpy + 16.) / 116.
    x = (a_numpy / 500.) + y
    z = y - (b_numpy / 200.)
    out = np.stack([x, y, z], axis=-1)
    mask_numpy = out > 0.2068966
    out[mask_numpy] = np.power(out[mask_numpy], 3.)
    out[~mask_numpy] = (out[~mask_numpy] - 16.0 / 116.) / 7.787
    out *= lab_ref_white
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    y_tensor = (l_tensor + torch.FloatTensor([16.])) / torch.FloatTensor([116.])
    x_tensor = (a_tensor / torch.FloatTensor([500.])) + y_tensor
    z_tensor = torch.clamp(y_tensor - (b_tensor / torch.FloatTensor([200.])), min=0.0)
    concat_xyz_tensor = torch.stack((x_tensor, y_tensor, z_tensor), dim=-1)

    xyz_mask = (concat_xyz_tensor > torch.FloatTensor([0.2068966])).bool()
    xyz_t = torch.mul(torch.pow(concat_xyz_tensor, 3), xyz_mask)
    xyz_f = torch.mul((concat_xyz_tensor - torch.FloatTensor([16.0]) / torch.FloatTensor([116.])) / torch.FloatTensor([7.787]), ~xyz_mask)
    xyz_tensor = torch.mul(xyz_t + xyz_f, torch.FloatTensor(lab_ref_white))

    rgb_from_xyz_tensor = torch.matmul(xyz_tensor, torch.FloatTensor(rgb_from_xyz.T))
    rgb_mask = (rgb_from_xyz_tensor > torch.FloatTensor([0.0031308])).bool()
    rgb_t = torch.mul(torch.FloatTensor([1.055]) * torch.pow(rgb_from_xyz_tensor, 1/2.4) - torch.FloatTensor([0.055]), rgb_mask)
    rgb_f = torch.mul(rgb_from_xyz_tensor * torch.FloatTensor([12.92]), ~rgb_mask)
    rgb_tensor = torch.clamp(rgb_t + rgb_f, min=0.0, max=1.0)
