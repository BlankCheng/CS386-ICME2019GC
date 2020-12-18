import os

import torch
from PIL import Image
from torchvision import transforms

from dataloader import preprocess
from model import Model

os.environ["VISIBLE_DEVICES"] = ''

loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()
root_path = '/NAS2020/Share/chenxianyu/PycharmProjects/CS386-ICME2019GC'
save_path = os.path.join(root_path, './checkpoints')


def save_image(tensor, name):
    dir = 'results'
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    if not os.path.exists(dir):
        os.makedirs(dir)
    image.save(name)


if __name__ == '__main__':
    best_epoch, phase, idx = 26, 'test', '272'
    image_path = os.path.join(root_path, './data/Images/{}/{}.png'.format(phase, idx))
    smap_path = os.path.join(root_path, './data/ASD_FixMaps/{}/{}_s.png'.format(phase, idx))
    img = Image.open(image_path)
    smap = Image.open(smap_path)
    img = preprocess(img)
    smap = preprocess(smap)
    img = torch.unsqueeze(img, 0)
    print(img.size())

    net = Model(input_size=(3, 224, 224))
    net.load_state_dict(torch.load(os.path.join(save_path, 'best_{}.pth'.format(best_epoch))))
    smap_pred = net(img)
    save_image(smap, '{}_gt.png'.format(idx))
    save_image(smap_pred, '{}_pred.png'.format(idx))
    print("Done.")
