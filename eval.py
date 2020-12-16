import argparse

import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as DT
from model import Model
from tqdm import tqdm

from dataloader import EyeTracker
from utils import *


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='/home/zhoujun/CS386-ICME2019GC')
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
    net = Model(input_size=(3, 224, 224))
    criterion = nn.MSELoss()
    net = net.to(device)
    if args.model_path:
        net.load_state_dict(torch.load(args.model_path))
    else:
        raise AssertionError("--model_path is not provided.")

    # inference
    smaps, smaps_pred = None, None
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
            if smaps is None:
                smaps = smap
                smaps_pred = smap_pred
            else:
                smaps = torch.cat((smaps, smap), dim=0)
                smaps_pred = torch.cat((smaps_pred, smap_pred), dim=0)
        print("test loss:{:.4f}".format(test_loss / i))

    # metrics
    sauc = calculate_sauc(smaps, smaps_pred)
    auc_j = calculate_auc_j(smaps, smaps_pred)
    nss = calculate_nss(smaps, smaps_pred)
    cc = calculate_cc(smaps, smaps_pred)
    sim = calculate_sim(smaps, smaps_pred)
    kld = calculate_kld(smaps, smaps_pred)
    print("sauc:{:.4f} | auc_j:{:.4f} | nss:{:.4f} | cc:{:.4f} | sim:{:.4f} | kld:{:.4f}".format(sauc, auc_j, nss, cc, sim, kld))