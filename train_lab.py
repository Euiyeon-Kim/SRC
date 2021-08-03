import os
import shutil

import cv2
import numpy as np
from skimage.color import lab2rgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_ssim
from config import Config as _config
from model.cielab_edsr import LABEDSR
from dataloader.cielab_df2k import DF2KLabOnlyTrain, DF2KLabOnlyValid
from utils import tv_lab2xyz, tv_xyz2rgb


def get_psnr(img1, img2, min_value=0, max_value=255):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = max_value - min_value
    return 10 * torch.log10((PIXEL_MAX ** 2) / mse)


def train(config):
    os.makedirs(f'exps/{config.exp_dir}/logs', exist_ok=True)
    os.makedirs(f'exps/{config.exp_dir}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{config.exp_dir}/samples', exist_ok=True)
    writer = SummaryWriter(f'exps/{config.exp_dir}/logs')
    shutil.copy('config.py', f'exps/{config.exp_dir}/config.py')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = DF2KLabOnlyTrain(config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)
    valid_dataset = DF2KLabOnlyValid(config)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False)

    model = LABEDSR(config).to(device)
    criterion = nn.L1Loss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)

    max_valid_psnr = -np.inf
    num_batch = len(train_loader)
    num_epoch = int(np.ceil(config.num_iters / len(train_dataset)))
    for epoch in range(num_epoch):
        for step, data in enumerate(train_loader):
            lr_lab, hr_lab = data
            lr_lab = lr_lab.to(device)
            hr_lab = hr_lab.to(device)

            optimizer.zero_grad()
            _, _, _, sr_lab = model(lr_lab)
            loss = criterion(sr_lab, hr_lab)
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/l1', loss.item(), (epoch * num_batch) + step)
            writer.flush()
            print(f'EPOCH {epoch} [{step}|{num_batch}]: {loss.item()}')

        scheduler.step()
        hr_img = lab2rgb(np.transpose(hr_lab[0].cpu().numpy(), (1, 2, 0))) * 255.
        lr_img = lab2rgb(np.transpose(lr_lab[0].cpu().numpy(), (1, 2, 0))) * 255.
        sr_img = lab2rgb(np.transpose(sr_lab[0].detach().cpu().numpy(), (1, 2, 0))) * 255.

        cv2.imwrite(f'exps/{config.exp_dir}/samples/{epoch}_hr.png', cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'exps/{config.exp_dir}/samples/{epoch}_lr.png', cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'exps/{config.exp_dir}/samples/{epoch}_sr.png', cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))

        model.eval()
        psnr = 0
        ssim = 0
        for _, data in enumerate(valid_loader):
            lr, hr = data
            lr = lr.to(device)
            hr = hr.to(device)
            l, a, b, _ = model(lr)
            sr_rgb = tv_xyz2rgb(tv_lab2xyz(l, a, b, device), device)
            psnr += get_psnr(sr_rgb, hr).item()
            ssim += pytorch_ssim.ssim(sr_rgb, hr).item()

        print(f'EPOCH {epoch}: PSNR:{psnr / len(valid_loader)} SSIM:{ssim / len(valid_loader)}')
        writer.add_scalar('valid/psnr', psnr / len(valid_loader), epoch)
        writer.add_scalar('valid/ssim', ssim / len(valid_loader), epoch)
        writer.flush()


if __name__ == '__main__':
    train(_config)
