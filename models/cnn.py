# -*- coding: utf-8 -*-
# @FileName: cnn.py
# Version: 0.0.1
# @Project: home_monitor
# @Author: Finebit
import time
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))  # imgSize=320*180 cnnOutput=77*42
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, (5, 5))
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(21888, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x shape 320 180
        x = self.pool(nnf.relu(self.bn1(self.conv1(x))))
        x = self.pool(nnf.relu(self.bn2(self.conv2(x))))
        x = self.pool(nnf.relu(self.bn3(self.conv3(x))))
        x = x.view(x.shape[0], -1)  # (batch, 16 * 77 * 77)
        x = nnf.relu(self.fc1(x))
        x = nnf.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    net = Net()
    a = torch.ones(1, 3, 180, 320)
    t1 = time.time()
    b = net(a)
    print(time.time() - t1)
    print(b)
