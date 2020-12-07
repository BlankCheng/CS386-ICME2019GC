import argparse
import logging
import os

import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as DT
from tqdm import tqdm

from dataloader import EyeTracker, MIT1003
from model import Model


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='D:\\zjcheng\\Workspace\\College\\专业课\\数字图像处理\\DIP2\\CS386-ICME2019GC')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['EyeTracker', 'MIT1003'], default='EyeTracker')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=str, default='1e-3')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        filename='train_log.txt',
                        filemode='a',
                        )
    logger = logging.getLogger()
    args = arg_parse()
    if args.seed != -1:
        torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    save_path = os.path.join(args.root_path, args.save_path)

    # load data
    if args.dataset == 'EyeTracker':
        train_dataset = EyeTracker(args=args, phase='train')
        val_dataset = EyeTracker(args=args, phase='val')
    elif args.dataset == 'MIT1003':
        train_dataset = MIT1003(args=args, phase='train')
        val_dataset = MIT1003(args=args, phase='val')
    else:
        raise NotImplementedError("{} is not supported now. Choose from [EyeTracker, MIT1003]".format(args.dataset))
    train_dataloader = DT.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False
    )
    val_dataloader = DT.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    # config network
    net = Model(input_size=(3, 224, 224))
    # net = nn.DataParallel(net)
    optimizer = optim.Adam(net.parameters(), lr=eval(args.lr), betas=(0.9, 0.999))
    criterion = nn.MSELoss()
    net = net.to(device)
    if args.model_path is not None:
        net.load_state_dict(torch.load(args.model_path))

    # train
    best_val_loss = 1000000
    print("-----------------Training Start-----------------\n")
    for epoch in range(1, args.max_epochs + 1):
        i, train_loss = 0, 0.0
        net.train()
        for data in tqdm(train_dataloader):
            img, smap = data[0].to(device), data[1].to(device)  # (b, 3, 224, 224), (b, 1, 224, 224)
            optimizer.zero_grad()
            smap_pred = net(img)  # (b, 1, 224, 224)
            loss = criterion(smap_pred, smap)
            loss.backward()
            optimizer.step()
            train_loss += loss
            i += 1
            print("#epoch:{}, #batch:{}, loss:{:.4f}".format(epoch, i, loss))
        torch.save(net.state_dict(), os.path.join(save_path, 'epoch_{}.pth'.format(epoch)))
        print("-----------------Validating Start-----------------\n")
        j, val_loss = 0, 0.0
        net.eval()
        with torch.no_grad():
            for data in tqdm(val_dataloader):
                img, smap = data[0].to(device), data[1].to(device)
                smap_pred = net(img)
                loss = criterion(smap_pred, smap)
                val_loss += loss
                j += 1
            print("validating loss:{:.4f}".format(val_loss / j))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New best model saved.")
                torch.save(net.state_dict(), os.path.join(save_path, 'best_{}.pth'.format(epoch)))
                # TODO: 计算具体指标
            logger.info("#epoch:{}, train loss:{:.4f}, val loss:{:.4f}".format(epoch, train_loss / i, val_loss / j))
            print("--------------------------------------------------\n")
