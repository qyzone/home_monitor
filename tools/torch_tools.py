# -*- coding: utf-8 -*-
# @FileName: torch_tools.py
# Version: 0.0.1
# @Project: home_monitor
# @Author: Finebit
import time
import torch


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


if __name__ == '__main__':
    pass
