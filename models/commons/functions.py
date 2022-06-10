# -*- coding: utf-8 -*-
# @FileName: functions.py
# Version: 0.0.1
# @Project: home_monitor
# @Author: Finebit
import torch
from torch import nn
from torch.nn import functional as nnf


def focus(x):
    if x.shape[-1] % 2 == 1:
        x = nnf.pad(x, [0, 1])
    if x.shape[-2] % 2 == 1:
        x = nnf.pad(x, (0, 0, 0, 1))
    x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
    return x


if __name__ == '__main__':
    # 偶偶
    a = torch.Tensor([[[[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]],
                       [[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8]],
                       [[9, 10, 9, 10], [11, 12, 11, 12], [9, 10, 9, 10], [11, 12, 11, 12]]]])
    # 偶奇
    b = torch.Tensor([[[[1, 2, 1, 2, 1], [3, 4, 3, 4, 3], [1, 2, 1, 2, 1], [3, 4, 3, 4, 3]],
                       [[5, 6, 5, 6, 5], [7, 8, 7, 8, 7], [5, 6, 5, 6, 5], [7, 8, 7, 8, 7]],
                       [[9, 10, 9, 10, 9], [11, 12, 11, 12, 11], [9, 10, 9, 10, 9], [11, 12, 11, 12, 11]]]])
    # 奇偶
    c = torch.Tensor([[[[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2]],
                       [[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6]],
                       [[9, 10, 9, 10], [11, 12, 11, 12], [9, 10, 9, 10], [11, 12, 11, 12], [9, 10, 9, 10]]]])
    # 奇奇
    d = torch.Tensor([[[[1, 2, 1, 2, 1], [3, 4, 3, 4, 3], [1, 2, 1, 2, 1], [3, 4, 3, 4, 3], [1, 2, 1, 2, 1]],
                       [[5, 6, 5, 6, 5], [7, 8, 7, 8, 7], [5, 6, 5, 6, 5], [7, 8, 7, 8, 7], [5, 6, 5, 6, 5]],
                       [[9, 10, 9, 10, 9], [11, 12, 11, 12, 11], [9, 10, 9, 10, 9], [11, 12, 11, 12, 11], [9, 10, 9, 10, 9]]]])

    i = d
    print(i.shape)
    o = focus(i)
    print(o.shape)
    print(o[0])
    pass