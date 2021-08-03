import os
import shutil

import cv2
import numpy as np
from skimage.color import lab2rgb

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pytorch_ssim
from utils import tv_rgb2xyz, tv_xyz2lab
from config import Config as _config
from model.label_lab_edsr import LabelLabEDSR
from dataloader.cielab_df2k import DF2KLabTrain, DF2KLabValid


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
    train_dataset = DF2KLabTrain(config)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)
    valid_dataset = DF2KLabValid(config)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=2, shuffle=False)

    model = LabelLabEDSR(config).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    mae = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.5)

    max_valid_psnr = -np.inf
    num_batch = len(train_loader)
    num_epoch = int(np.ceil(config.num_iters / len(train_dataset)))
    for epoch in range(num_epoch):
        model.train()
        for step, data in enumerate(train_loader):
            lr_imgs, hr_labels, hr_labs = data
            lr_imgs = lr_imgs.to(device)
            hr_labels = hr_labels.to(device)
            hr_labs = hr_labs.to(device)

            optimizer.zero_grad()
            sr_pred = model(lr_imgs)
            classification_loss = criterion(sr_pred, hr_labels)
            sr_img = torch.argmax(nn.functional.softmax(sr_pred, dim=1), dim=1).float()
            sr_labs = tv_xyz2lab(tv_rgb2xyz(sr_img, device), device)
            mae_loss = mae(sr_labs, hr_labs)
            loss = classification_loss + mae_loss
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/CrossEntropy', classification_loss.item(), (epoch * num_batch) + step)
            writer.add_scalar('train/MAE', mae_loss.item(), (epoch * num_batch) + step)
            writer.add_scalar('train/total', loss.item(), (epoch * num_batch) + step)
            writer.flush()
            print(f'EPOCH {epoch} [{step}|{num_batch}]: {loss.item()}, MAE:{mae_loss.item()}, CE:{classification_loss.item()}')
            break

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

        if max_valid_psnr < psnr:
            max_valid_psnr = psnr
            torch.save(model.state_dict(), f'exps/{config.exp_dir}/ckpt/{psnr:.4f}.pth')

        scheduler.step()
        viz_pred = sr_img[0]
        viz_lab = sr_labs[0].cpu().numpy()
        viz_rgb = lab2rgb(viz_lab)
        save_image((hr_labels[0]) / 255., f'exps/{config.exp_dir}/samples/{epoch}_hr.png')
        save_image((lr_imgs[0] + 1) / 2., f'exps/{config.exp_dir}/samples/{epoch}_lr.png')
        save_image(viz_pred / 255., f'exps/{config.exp_dir}/samples/{epoch}_sr.png')
        cv2.imwrite(f'exps/{config.exp_dir}/samples/{epoch}_recon.png', cv2.cvtColor(viz_rgb, cv2.COLOR_RGB2BGR)*255.)


if __name__ == '__main__':
    train(_config)
