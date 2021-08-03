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
from model.label_edsr import LabelEDSR
from dataloader.rgb_df2k import DF2KLabelTrain, DF2KLabelValid


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
    train_dataset = DF2KLabelTrain(config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)
    valid_dataset = DF2KLabelValid(config)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=2, shuffle=False)

    model = LabelEDSR(config).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)

    max_valid_err = np.inf
    num_batch = len(train_loader)
    num_epoch = int(np.ceil(config.num_iters / len(train_dataset)))
    for epoch in range(num_epoch):
        model.train()
        for step, data in enumerate(train_loader):
            lr_imgs, hr_labels = data
            lr_imgs = lr_imgs.to(device)
            hr_labels = hr_labels.to(device)

            optimizer.zero_grad()
            sr_pred = model(lr_imgs)
            loss = criterion(sr_pred, hr_labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/CrossEntropy', loss.item(), (epoch * num_batch) + step)
            writer.flush()
            print(f'EPOCH {epoch} [{step}|{num_batch}]: {loss.item()}')

        model.eval()
        psnr = 0
        ssim = 0
        for _, data in enumerate(valid_loader):
            lr, hr = data
            lr = lr.to(device)
            hr = hr.to(device)
            sr_pred = model(lr)
            sr_img = torch.argmax(nn.functional.softmax(sr_pred, dim=1), dim=1).float()
            psnr += get_psnr(sr_img, hr).item()
            ssim += pytorch_ssim.ssim(sr_img, hr).item()

        print(f'EPOCH {epoch}: PSNR:{psnr/len(valid_loader)} SSIM:{ssim/len(valid_loader)}')
        writer.add_scalar('valid/psnr', psnr / len(valid_loader), epoch)
        writer.add_scalar('valid/ssim', ssim / len(valid_loader), epoch)
        writer.flush()

        if psnr < max_valid_err:
            max_valid_err = psnr
            torch.save(model.state_dict(), f'exps/{config.exp_dir}/ckpt/{psnr:.4f}.pth')

        scheduler.step()
        viz_pred = torch.argmax(nn.functional.softmax(sr_pred[0], dim=0), dim=0)
        save_image((hr_labels[0]) / 255., f'exps/{config.exp_dir}/samples/{epoch}_hr.png')
        save_image((lr_imgs[0] + 1) / 2., f'exps/{config.exp_dir}/samples/{epoch}_lr.png')
        save_image(viz_pred / 255., f'exps/{config.exp_dir}/samples/{epoch}_sr.png')


if __name__ == '__main__':
    train(_config)
