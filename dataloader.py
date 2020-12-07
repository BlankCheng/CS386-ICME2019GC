import os

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as DT
from PIL import Image
from torchvision import transforms

np.set_printoptions(threshold=np.inf)
CLIP_THRES = 0.05
WIDTH = HEIGHT = 224
preprocess = transforms.Compose([
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
    transforms.Resize((WIDTH, HEIGHT)),
    transforms.ToTensor(),  # [0,255]的np变成[0,1]的tensor
])
augment = transforms.Compose([
    # transforms.RandomSizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(brightness=0., contrast=1, saturation=0., hue=0.),
    # transforms.RandomAffine(20, translate=(0, 0.2), scale=(0.9, 1), shear=1, resample=False, fillcolor=0),
])


class EyeTracker(DT.Dataset):
    def __init__(self, args, phase):
        super(EyeTracker, self).__init__()
        self.phase = phase
        self.img_path = os.path.join(args.root_path, args.data_path, "Images", self.phase)
        self.smap_path = os.path.join(args.root_path, args.data_path, "ASD_FixMaps", self.phase)
        self.img_files = os.listdir(self.img_path)
        self.imgs, self.smaps = [], []  # list of PILs
        for file in self.img_files:
            img = preprocess(Image.open(os.path.join(self.img_path, file)))
            smap = preprocess(Image.open(os.path.join(self.smap_path, file.split('.')[0] + '_s' + '.png')))
            smap[(smap < CLIP_THRES)] = 0.0
            self.imgs.append(img)
            self.smaps.append(smap)

    def __getitem__(self, index):
        img = self.imgs[index]
        smap = self.smaps[index]
        if self.phase == 'train':
            img = augment(img)  # TODO: augment需要再img和smap上作用相同
            smap = augment(smap)
        return img, smap

    def __len__(self):
        return len(self.imgs)


class MIT1003(DT.Dataset):
    # TODO
    def __init__(self, args, phase):
        super(MIT1003, self).__init__()
        self.data = [i for i in range(100)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    root_path = 'D:\\zjcheng\\Workspace\\College\\专业课\\数字图像处理\\DIP2\\CS386-ICME2019GC'
    img = Image.open(os.path.join(root_path, 'data/Images/train/1.png'))
    img = transforms_train(img)
    smap = Image.open(os.path.join(root_path, 'data/ASD_FixMaps/train/1_s.png'))
    smap = transforms_train(smap)
    plt.figure("Image")
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.imshow(img)
    plt.waitforbuttonpress()
    plt.imshow(smap)
    plt.show()
