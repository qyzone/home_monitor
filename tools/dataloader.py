# -*- coding: utf-8 -*-
# @FileName: dataloader.py
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
    def __init__(self, size):  # h w
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            to_tensor = transforms.ToTensor()  # c h w
            img = to_tensor(img)
        img_shape = img.shape[-2:]
        r = min(self.size[0] / img_shape[0], self.size[1] / img_shape[1])
        new_unpad = [int(round(img_shape[0] * r)), int(round(img_shape[1] * r))]
        img = transforms.functional.resize(img, new_unpad)
        dh, dw = self.size[0] - new_unpad[0], self.size[1] - new_unpad[1]  # wh padding
        dh, dw = dh / 2, dw / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = nnf.pad(img, (left, right, top, bottom), value=0)  # 倒序给值 w h c
        return img


def load_images(root, size, batch_size=1):
    img_transform = transforms.Compose([Resize(size), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = ImageFolder(root, transform=img_transform)
    # dataset = dataset.to(device)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)  #
    return data_loader


if __name__ == '__main__':
    a = torch.ones(1, 3, 128, 64)
    b = torch.ones(1, 3, 64, 128)
    resize = Resize((32, 32))
    c = resize(a)
    print(c.shape)
