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

from config import Config as _config
from model.cielab_edsr import LABEDSR
from dataloader.cielab_df2k import DF2KLAB


def train(config):
    os.makedirs(f'exps/{config.exp_dir}/logs', exist_ok=True)
    os.makedirs(f'exps/{config.exp_dir}/ckpt', exist_ok=True)
    os.makedirs(f'exps/{config.exp_dir}/samples', exist_ok=True)
    writer = SummaryWriter(f'exps/{config.exp_dir}/logs')
    shutil.copy('config.py', f'exps/{config.exp_dir}/config.py')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = DF2KLAB(config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)

    model = LABEDSR(config).to(device)
    criterion = nn.L1Loss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)

    num_batch = len(train_loader)
    num_epoch = int(np.ceil(config.num_iters / len(train_dataset)))
    for epoch in range(num_epoch):
        for step, data in enumerate(train_loader):
            lr_lab, hr_lab, rgb_label = data
            lr_lab = lr_lab.to(device)
            hr_lab = hr_lab.to(device)
            # rgb_label = rgb_label.to(device)

            optimizer.zero_grad()
            l_tensor, a_tensor, b_tensor, sr_lab = model(lr_lab)
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


if __name__ == '__main__':
    train(_config)
