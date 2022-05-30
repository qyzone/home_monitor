# -*- coding: utf-8 -*-
# @FileName: resnet.py
# Version: 0.0.1
# @Project: home_monitor
# @Author: Finebit
import time
import torch
import torch.nn.functional as nnf
from torch import nn


# 定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=(1, 1)):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(3, 3), stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),  # stride=1
            nn.BatchNorm2d(c_out)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(c_out)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = nnf.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.c_in = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=2)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)  # 512

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.c_in, channels, stride))
            self.c_in = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nnf.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    net = ResNet18(num_classes=2)
    a = torch.ones(1, 3, 320, 180)
    t1 = time.time()
    b = net(a)
    print(time.time() - t1)
    print(b)
