# -*- coding: utf-8 -*-
# @FileName: focusnet.py
# Version: 0.0.1
# @Project: home_monitor
# @Author: Finebit
import torch
from torch import nn
import torch.nn.functional as nnf
from models.commons.blocks import ResidualBlock
from models.commons.functions import focus
import time


class FocusNet(nn.Module):
    def __init__(self):
        super(FocusNet, self).__init__()
        self.layer1 = ResidualBlock(3, 3)
        self.layer2 = ResidualBlock(12, 12)
        self.layer3 = ResidualBlock(48, 48)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(192, 24)
        self.bn = nn.BatchNorm1d(24)
        self.fc2 = nn.Linear(24, 2)

    def forward(self, x):
        x = focus(self.layer1(x))
        x = focus(self.layer2(x))
        x = focus(self.layer3(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = nnf.relu(self.bn(self.fc1(x)), inplace=True)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    net = FocusNet()
    a = torch.ones(1, 3, 180, 320)
    for i in range(10):
        t1 = time.time()
        with torch.no_grad():
            net.eval()
            b = net(a)
        print(b)
        print(time.time() - t1)

    # a = torch.Tensor([[[[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]],
    #                    [[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8]],
    #                    [[9, 10, 9, 10], [11, 12, 11, 12], [9, 10, 9, 10], [11, 12, 11, 12]]]])
    # print(nnf.softmax(a, dim=3))
    # print(nnf.softmax(a, dim=-1))
