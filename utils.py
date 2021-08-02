import torch
import numpy as np
from scipy import linalg

LAB_WHITE = (0.95047, 1., 1.08883)
XYZ_FROM_RGB = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])
RGB_FROM_XYZ = linalg.inv(XYZ_FROM_RGB)


def lab2rgb(l_tensor, a_tensor, b_tensor):
    y_tensor = (l_tensor + torch.FloatTensor([16.])) / torch.FloatTensor([116.])
    x_tensor = (a_tensor / torch.FloatTensor([500.])) + y_tensor
    z_tensor = torch.clamp(y_tensor - (b_tensor / torch.FloatTensor([200.])), min=0.0)
    concat_xyz_tensor = torch.stack((x_tensor, y_tensor, z_tensor), dim=-1)

    xyz_mask = (concat_xyz_tensor > torch.FloatTensor([0.2068966])).bool()
    xyz_t = torch.mul(torch.pow(concat_xyz_tensor, 3), xyz_mask)
    xyz_f = torch.mul(
        (concat_xyz_tensor - torch.FloatTensor([16.0]) / torch.FloatTensor([116.])) / torch.FloatTensor([7.787]),
        ~xyz_mask)
    xyz_tensor = torch.mul(xyz_t + xyz_f, torch.FloatTensor(LAB_WHITE))

    rgb_from_xyz_tensor = torch.matmul(xyz_tensor, torch.FloatTensor(RGB_FROM_XYZ.T))
    rgb_mask = (rgb_from_xyz_tensor > torch.FloatTensor([0.0031308])).bool()
    rgb_t = torch.mul(torch.FloatTensor([1.055]) * torch.pow(rgb_from_xyz_tensor, 1 / 2.4) - torch.FloatTensor([0.055]),
                      rgb_mask)
    rgb_f = torch.mul(rgb_from_xyz_tensor * torch.FloatTensor([12.92]), ~rgb_mask)
    rgb_tensor = torch.clamp(rgb_t + rgb_f, min=0.0, max=1.0)
    return rgb_tensor
