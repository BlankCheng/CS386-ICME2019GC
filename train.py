import argparse
import logging
import os

import torch.optim as optim
import torch.utils.data as DT
from tqdm import tqdm

from dataloader import EyeTracker, MIT1003
from model import Model
from utils import *


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='/NAS2020/Share/chenxianyu/PycharmProjects/CS386-ICME2019GC')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['EyeTracker', 'MIT1003'], default='EyeTracker')
    parser.add_argument('--alpha1', type=str, default='0')
    parser.add_argument('--alpha2', type=str, default='0')
    parser.add_argument('--alpha3', type=str, default='0')
    parser.add_argument('--alpha4', type=str, default='0')
    parser.add_argument('--weight_decay', type=str, default='1e-4')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=str, default='1e-3')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        filename='train_log.txt',
                        filemode='a',
                        )
    logger = logging.getLogger()
    args = arg_parse()
    if args.seed != -1:
        torch.manual_seed(args.seed)
    print(torch.cuda.is_available())
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
    '''net = Model(input_size=(3, 224, 224), encoder_name='densenet169', extract_list=['denseblock1', 'denseblock2'],
                channels=[256, 512])'''
    net = Model(input_size=(3, 224, 224), encoder_name='resnet101', extract_list=['layer2', 'layer3', 'layer4'],
                channels=[512, 1024, 2048])
    '''net = Model(input_size=(3, 224, 224), encoder_name='vgg16', extract_list=["15", "22"],
                channels=[256, 512])'''
    '''net = Model(input_size=(3, 224, 224), encoder_name='se_resnext101', extract_list=['layer2', 'layer3', 'layer4'],
                channels=[512, 1024, 2048])'''
    # net = nn.DataParallel(net)
    optimizer = optim.Adam(net.parameters(), lr=eval(args.lr), betas=(0.9, 0.999), weight_decay=eval(args.weight_decay))
    bce = nn.BCELoss()
    net = net.to(device)
    if args.model_path is not None:
        net.load_state_dict(torch.load(args.model_path))

    # train
    best_val_loss = 1000000
    alpha1, alpha2, alpha3, alpha4 = eval(args.alpha1), eval(args.alpha2), eval(args.alpha3), eval(args.alpha4)
    print("-----------------Training Start-----------------\n")
    for epoch in range(1, args.max_epochs + 1):
        i, train_loss = 0, 0.0
        net.train()
        for data in tqdm(train_dataloader):
            img, smap, fmap = data[0].to(device), data[1].to(device), data[2]  # (b, 3, 224, 224), (b, 1, 224, 224)
            optimizer.zero_grad()
            smap_pred = net(img)  # (b, 1, 224, 224)
            loss = bce(smap_pred, smap) \
                   - alpha1 * calculate_nss(smap, smap_pred) \
                   - alpha2 * calculate_cc(smap, smap_pred) \
                   - alpha3 * calculate_sim(smap, smap_pred) \
                   + alpha4 * calculate_kld(smap, smap_pred)
            loss.backward()
            optimizer.step()
            train_loss += loss
            i += 1
            print("#epoch:{}, #batch:{}, loss:{:.4f}".format(epoch, i, loss))
        # torch.save(net.state_dict(), os.path.join(save_path, 'epoch_{}.pth'.format(epoch)))
        # validate
        print("-----------------Validating Start-----------------\n")
        j, val_loss = 0, 0.0
        smaps, smaps_pred = None, None
        net.eval()
        with torch.no_grad():
            for data in tqdm(val_dataloader):
                img, smap, fmap = data[0].to(device), data[1].to(device), data[2].to(device)  # (b, 3, 224, 224), (b, 1, 224, 224)
                smap_pred = net(img)
                loss = bce(smap_pred, smap) \
                       - alpha1 * calculate_nss(smap, smap_pred) \
                       - alpha2 * calculate_cc(smap, smap_pred) \
                       - alpha3 * calculate_sim(smap, smap_pred) \
                       + alpha4 * calculate_kld(smap, smap_pred)
                val_loss += loss
                j += 1
                if smaps is None:
                    smaps = smap
                    smaps_pred = smap_pred
                else:
                    smaps = torch.cat((smaps, smap), dim=0)
                    smaps_pred = torch.cat((smaps_pred, smap_pred), dim=0)
            print("validating loss:{:.4f}".format(val_loss / j))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("New best model saved.")
                torch.save(net.state_dict(), os.path.join(save_path, 'w_resnext_best_{}.pth'.format(epoch)))
            # metrics
            sauc = -1  # calculate_sauc(smaps, smaps_pred)
            auc_j = -1  # calculate_auc_j(smaps, smaps_pred)
            nss = calculate_nss(smaps, smaps_pred)
            cc = calculate_cc(smaps, smaps_pred)
            sim = calculate_sim(smaps, smaps_pred)
            kld = calculate_kld(smaps, smaps_pred)
            print(
                "sauc:{:.4f} | auc_j:{:.4f} | nss:{:.4f} | cc:{:.4f} | sim:{:.4f} | kld:{:.4f}".format(sauc, auc_j, nss,
                                                                                                       cc,
                                                                                                       sim, kld))
            logger.info("#epoch:{}, train loss:{:.4f}, val loss:{:.4f}\n".format(epoch, train_loss / i, val_loss / j))
            logger.info(
                "#epoch:{}, sauc:{:.4f} | auc_j:{:.4f} | nss:{:.4f} | cc:{:.4f} | sim:{:.4f} | kld:{:.4f}\n".format(
                    epoch, sauc, auc_j, nss, cc, sim, kld))
            print("--------------------------------------------------\n")
