import argparse
from tqdm import tqdm
import os
import torch.nn.parallel
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as DT
import pandas as pd
from sklearn.metrics import *
import numpy as np
# import torchvision.models as models

from dataloader import EyeTracker, MIT1003
from models.resnet import *  # TODO: 修改文件名


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='D:\\zjcheng\\Workspace\\College\\专业课\\数字图像处理\\DIP2\\CS386-ICME2019GC')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['EyeTracker', 'MIT1003'], default='EyeTracker')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = arg_parse()
    if args.seed != -1:
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    # load data
    if args.dataset == 'EyeTracker':
        test_dataset = EyeTracker(args=args, phase='test')
        val_dataset = EyeTracker(args=args, phase='val')
    else:
        raise NotImplementedError("{} is not supported now. Choose from [EyeTracker, MIT1003]".format(args.dataset))
    test_dataloader = DT.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    # config network
    net = resnet152(num_classes=10)  # TODO: 修改模型名
    criterion = nn.MSELoss()
    net = net.to(device)
    if args.model_path:
        net.load_state_dict(torch.load(args.model_path))
    else:
        raise AssertionError("--model_path is not provided.")

    # inference
    smaps, smaps_pred = [], []
    i, test_loss = 0, 0.0
    net.eval()
    print("-----------------Testing Start-----------------\n")
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            img, smap = data[0].to(device), data[1].to(device)
            smap_pred = net(img)
            loss = criterion(smap_pred, smap)  # TODO
            test_loss += loss
            i += 1
            smaps.append(smap.detach().cpu().numpy().squeeze((0, 1)))
            smaps_pred.append(smap_pred.detach().cpu().numpy().squeeze((0, 1)))
        print("test loss:{:.4f}".format(test_loss / i))

    # eval
    # TODO: 具体指标
