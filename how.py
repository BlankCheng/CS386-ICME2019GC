import matplotlib.pyplot as plt
import os
from PIL import Image
from dataloader import preprocess

img = Image.open('val1.png')
root_path = '/NAS2020/Share/chenxianyu/PycharmProjects/CS386-ICME2019GC'
save_path = os.path.join(root_path, './checkpoints')
smap1 = Image.open(os.path.join(root_path, 'data/ASD_FixMaps/val/241_s.png'))
smap1 = preprocess(smap1)
plt.figure("Image")  # 图像窗口名称
plt.imshow(smap1[0], cmap='gray')
plt.axis('on')  # 关掉坐标轴为 off
plt.title('image')  # 图像题目
plt.show()
