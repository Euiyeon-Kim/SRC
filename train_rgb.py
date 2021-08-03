import os
import shutil

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_ssim
from config import Config as _config
from model.rgb_edsr import EDSR
from dataloader.rgb_df2k import DF2KRGBTrain, DF2KRGBValid


def get_psnr(img1, img2, min_value=-1, max_value=1):
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
    train_dataset = DF2KRGBTrain(config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)
    valid_dataset = DF2KRGBValid(config)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False)

    model = EDSR(config).to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)

    valid_err = np.inf
    num_batch = len(train_loader)
    num_epoch = int(np.ceil(config.num_iters / len(train_dataset)))
    for epoch in range(num_epoch):
        model.train()
        for step, data in enumerate(train_loader):
            lr_imgs, hr_imgs = data
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/l1', loss.item(), (epoch * num_batch) + step)
            writer.flush()
            print(f'EPOCH {epoch} [{step}|{num_batch}]: {loss.item()}')

        model.eval()
        psnr = 0
        ssim = 0
        for _, data in enumerate(valid_loader):
            lr, hr = data
            lr = lr.to(device)
            hr = hr.to(device)
            sr = model(lr)
            psnr += get_psnr(sr, hr).item()
            ssim += pytorch_ssim.ssim(sr, hr).item()

        print(f'EPOCH {epoch}: PSNR:{psnr/len(valid_loader)} SSIM:{ssim/len(valid_loader)}')
        writer.add_scalar('valid/psnr', psnr / len(valid_loader), epoch)
        writer.add_scalar('valid/ssim', ssim / len(valid_loader), epoch)
        writer.flush()

        scheduler.step()
        save_image((hr_imgs[0] + 1) / 2., f'exps/{config.exp_dir}/samples/{epoch}_hr.png')
        save_image((lr_imgs[0] + 1) / 2., f'exps/{config.exp_dir}/samples/{epoch}_lr.png')
        save_image((sr_imgs[0] + 1) / 2., f'exps/{config.exp_dir}/samples/{epoch}_sr.png')


if __name__ == '__main__':
    train(_config)
