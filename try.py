import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from dataloader import preprocess
from model import Model

loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()


def save_image(tensor, name):
    dir = 'results'
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    if not os.path.exists(dir):
        os.makedirs(dir)
    image.save(name)


if __name__ == '__main__':
    root_path = '/NAS2020/Share/chenxianyu/PycharmProjects/CS386-ICME2019GC'
    save_path = os.path.join(root_path, './checkpoints')
    img1 = Image.open(os.path.join(root_path, 'data/Images/train/1.png'))
    img1 = preprocess(img1)
    img1 = img1.reshape((1,) + img1.shape)
    print(img1.shape)
    img2 = Image.open(os.path.join(root_path, 'data/Images/val/241.png'))
    img2 = preprocess(img2)
    img2 = img2.reshape((1,) + img2.shape)
    print(img2.shape)
    smap1 = Image.open(os.path.join(root_path, 'data/ASD_FixMaps/train/1_s.png'))
    smap1 = preprocess(smap1)
    smap2 = Image.open(os.path.join(root_path, 'data/ASD_FixMaps/val/241_s.png'))
    smap2 = preprocess(smap2)
    net = Model(input_size=(3, 224, 224))
    net.load_state_dict(torch.load(os.path.join(save_path, 'epoch_{}.pth'.format(60))))
    result1 = net(img1)
    print(result1[0])
    result2 = net(img2)
    print(result2[0])
    print("Fuck")
    save_image(result1, 'train1.png')
    save_image(result2, 'val1.png')
    '''
    plt.figure(figsize=(10, 5))  # 设置窗口大小
    plt.suptitle('imgage')  # 图片名称
    plt.subplot(2, 3, 1), plt.title('img1')
    plt.imshow(img1), plt.axis('off')

    plt.subplot(2, 3, 2), plt.title('origin1')
    plt.imshow(smap1, cmap='gray'), plt.axis('off')  # 这里显示灰度图要加cmap

    plt.subplot(2, 3, 3), plt.title('img2')
    plt.imshow(img2), plt.axis('off')

    plt.subplot(2, 3, 4), plt.title('origin2')
    plt.imshow(smap2, cmap='gray'), plt.axis('off')

    plt.subplot(2, 3, 5), plt.title('result1')
    plt.imshow(result1, cmap='gray'), plt.axis('off')

    plt.subplot(2, 3, 6), plt.title('result2')
    plt.imshow(result2, cmap='gray'), plt.axis('off')

    plt.show()
    '''
