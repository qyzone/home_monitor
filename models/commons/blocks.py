# -*- coding: utf-8 -*-
# @FileName: blocks.py
# Version: 0.0.1
# @Project: home_monitor
# @Author: Finebit
from torch import nn
from torch.nn import functional as nnf


class BottleNeckBlock(nn.Module):
    def __init__(self, ch, expansion=2, k_size=(3, 3)):
        super().__init__()
        mid_ch = int(ch/expansion)
        self.conv1 = nn.Conv2d(ch, mid_ch, (1, 1), bias=False)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, k_size, bias=False)
        self.conv3 = nn.Conv2d(mid_ch, ch, (1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DepthWiseBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthWiseBlock, self).__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, (5, 5), groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, (1, 1))
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.bn(x)
        x = nnf.relu(x)
        return x


# 定义残差块ResBlock
class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=(1, 1)):
        super(ResidualBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=(3, 3), stride=stride, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=(3, 3), stride=(1, 1), padding=1),  # stride=1
            nn.BatchNorm2d(c_out)
        )
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or c_in != c_out:
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


if __name__ == '__main__':
    pass
