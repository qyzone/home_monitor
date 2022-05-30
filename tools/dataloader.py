# -*- coding: utf-8 -*-
# @FileName: dataloader.py
# Version: 0.0.1
# @Project: home_monitor
# @Author: Finebit

from torch import nn
from torch.nn import functional as nnf
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


class Resize(nn.Module):
    def __init__(self, size, device):  # h w
        self.device = device
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def forward(self, img):
        img_tensor = img
        if not isinstance(img, torch.Tensor):
            to_tensor = transforms.ToTensor()  # c h w
            img_tensor = to_tensor(img)
        # img_tensor = img_tensor.to(self.device)
        img_shape = img_tensor.shape[1:3]
        r = min(self.size[0] / img_shape[0], self.size[1] / img_shape[1])
        new_unpad = [int(round(img_shape[0] * r)), int(round(img_shape[1] * r))]
        new_img = transforms.functional.resize(img_tensor, new_unpad)
        dh, dw = self.size[0] - new_unpad[0], self.size[1] - new_unpad[1]  # wh padding
        dh, dw = dh / 2, dw / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        new_img = nnf.pad(new_img, (left, right, top, bottom), value=0)  # 倒序给值 w h c
        return new_img


def load_images(root, size, device, batch_size=1):
    img_transform = transforms.Compose([Resize(size, device), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = ImageFolder(root, transform=img_transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)  #
    return data_loader


if __name__ == '__main__':
    a = torch.ones(128, 64, 3)
    b = torch.ones(128, 64, 3)
    resize = Resize((32, 32), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
