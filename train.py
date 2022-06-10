# -*- coding: utf-8 -*-
# @FileName: train.py
# Version: 0.0.1
# @Project: home_monitor
# @Author: Finebit
import os.path
from pathlib import Path

import torch
from torch import optim
from torch import nn
from tools.dataloader import load_images
from models.focusnet import FocusNet as Net
# from models.cnn import Net


def train(checkpoint_dir, source: list, size, batch=16, epoch=50):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir() if (not checkpoint_dir.exists()) else True
    checkpoint_path = checkpoint_dir / "best.pth.tar"
    train_source = source[0]
    valid_source = source[1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    net = Net()
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, amsgrad=True)  # betas=(0.9, 0.999) eps=1e-08
    epoch_start = 0
    verify_loss = 1.0
    if os.path.exists(checkpoint_path):
        model_pth = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(model_pth['state_dict'])
        optimizer.load_state_dict(model_pth['optimizer'])
        epoch_start = model_pth['epoch']
        verify_loss = model_pth['best_loss']
    for par_grp in optimizer.param_groups:
        print(f'lr: {par_grp["lr"]}, betas: {par_grp["betas"]}, eps: {par_grp["eps"]}, amsgrad: {par_grp["amsgrad"]}')
    criterion = nn.CrossEntropyLoss()  # 包含log_softmax, one-hot
    train_loads = load_images(train_source, size=size, batch_size=batch)
    valid_loads = load_images(valid_source, size=size, batch_size=batch)
    train_len = len(train_loads)
    valid_len = len(valid_loads)
    for epoch in range(epoch_start, epoch):
        train_loss = 0.0
        valid_loss = 0.0
        for i, data in enumerate(train_loads):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss
            if i % 10 == 9:
                print(f'{i}/{train_len}')
            if i % 100 == 99:
                print('[%d, %5d] train loss: %.3f' % (epoch+1, i+1, train_loss/100))
                train_loss = 0.0
        print('[%d] train loss: %.3f' % (epoch + 1, train_loss/train_len))
        with torch.no_grad():
            net.eval()
            for i, data in enumerate(valid_loads):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                valid_loss += loss
                if i % 10 == 9:
                    print(f'{i}/{valid_len}')
            valid_loss = valid_loss / valid_len
            print('[%d] valid loss: %.3f' % (epoch + 1, valid_loss))
            if epoch <= 1 or verify_loss > valid_loss:
                verify_loss = valid_loss
                torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(), 'best_loss': verify_loss,
                            'optimizer': optimizer.state_dict()}, checkpoint_path)
                print(f'checkpoint saved in {checkpoint_path}')
            net.train()
    print('Finished training')


if __name__ == '__main__':
    torch.set_num_threads(1)
    # dataset_root = "imageFolderPerson"
    dataset_root = "cam_record"
    train_src = f"../datasets/{dataset_root}/train"
    valid_src = f"../datasets/{dataset_root}/valid"
    _checkpoint_dir = "runs/train"
    train(_checkpoint_dir, [train_src, valid_src], (180, 320), batch=24, epoch=200)
