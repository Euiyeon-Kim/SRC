import torch
import numpy as np
from scipy import linalg

LAB_WHITE = (0.95047, 1., 1.08883)
XYZ_FROM_RGB = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])
RGB_FROM_XYZ = linalg.inv(XYZ_FROM_RGB)


def tv_lab2xyz(l_tensor, a_tensor, b_tensor, device):
    y_tensor = (l_tensor + torch.FloatTensor([16.]).to(device)) / torch.FloatTensor([116.]).to(device)
    x_tensor = (a_tensor / torch.FloatTensor([500.]).to(device)) + y_tensor
    z_tensor = torch.clamp(y_tensor - (b_tensor / torch.FloatTensor([200.]).to(device)), min=0.0)
    concat_xyz_tensor = torch.stack((x_tensor, y_tensor, z_tensor), dim=-1)

    xyz_mask = (concat_xyz_tensor > torch.FloatTensor([0.2068966]).to(device)).bool()
    xyz_t = torch.mul(torch.pow(concat_xyz_tensor, 3), xyz_mask)
    xyz_f = torch.mul((concat_xyz_tensor - torch.FloatTensor([16.0]).to(device) / torch.FloatTensor([116.]).to(device))
                      / torch.FloatTensor([7.787]).to(device), ~xyz_mask)
    xyz_tensor = torch.mul(xyz_t + xyz_f, torch.FloatTensor(LAB_WHITE).to(device))
    return xyz_tensor


def tv_rgb2xyz(rgb_tensor, device):
    rgb_tensor = rgb_tensor.permute(0, 2, 3, 1) / torch.FloatTensor([255.]).to(device)
    xyz_mask = (rgb_tensor > torch.FloatTensor([0.04045]).to(device)).bool()
    xyz_t = torch.mul(rgb_tensor, xyz_mask) + torch.mul(torch.FloatTensor([0.055]).to(device), xyz_mask)
    xyz_t = torch.pow(xyz_t / torch.FloatTensor([1.055]).to(device), 2.4)
    xyz_f = torch.mul(rgb_tensor, ~xyz_mask) / torch.FloatTensor([12.92]).to(device)
    xyz_tensor = xyz_t + xyz_f
    recon = torch.matmul(xyz_tensor, torch.FloatTensor(XYZ_FROM_RGB.T).to(device))
    return recon


def tv_xyz2lab(xyz_tensor, device):
    xyz_tensor /= torch.FloatTensor(LAB_WHITE).to(device)
    lab_mask = (xyz_tensor > torch.FloatTensor([0.008856]).to(device)).bool()
    lab_t = torch.pow(torch.mul(xyz_tensor, lab_mask), 1/3)
    lab_f = torch.FloatTensor([7.787]).to(device) * xyz_tensor + torch.FloatTensor([16.]).to(device) / torch.FloatTensor([116.]).to(device)
    lab_f = torch.mul(lab_f, ~lab_mask)
    tmp = lab_t + lab_f
    x_tensor, y_tensor, z_tensor = torch.tensor_split(tmp, 3, dim=-1)
    l_tensor = torch.FloatTensor([116.]).to(device) * y_tensor - torch.FloatTensor([16.]).to(device)
    a_tensor = torch.FloatTensor([500.0]).to(device) * (x_tensor-y_tensor)
    b_tensor = torch.FloatTensor([200.0]).to(device) * (y_tensor - z_tensor)
    return torch.cat((l_tensor, a_tensor, b_tensor), dim=-1)


def tv_xyz2rgb(xyz_tensor, device):
    rgb_from_xyz_tensor = torch.matmul(xyz_tensor, torch.FloatTensor(RGB_FROM_XYZ.T).to(device))
    rgb_mask = (rgb_from_xyz_tensor > torch.FloatTensor([0.0031308]).to(device)).bool()
    rgb_t = torch.FloatTensor([1.055]).to(device) * torch.pow(torch.mul(rgb_from_xyz_tensor, rgb_mask), 1 / 2.4)
    rgb_t = rgb_t - torch.multiply(rgb_mask, torch.FloatTensor([0.055]).to(device))
    rgb_f = torch.mul(rgb_from_xyz_tensor, ~rgb_mask) * torch.FloatTensor([12.92]).to(device)
    rgb_tensor = torch.clamp(rgb_t + rgb_f, min=0.0, max=1.0)
    return rgb_tensor