import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from glob import glob

dir_data = 'E:\dataset'
dir_train = os.path.join(dir_data, 'train')
dir_val = os.path.join(dir_data, 'val')
dir_test = os.path.join(dir_data, 'test')

class kits19_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, mode='kid'):   
        # data_dir 자리에는 train | val | test가 온다
        # mode에는 all | kid | tum가 오며, all은 모든 슬라이스에 대해, kid와 tum는 각각 kidney와 tumor가 존재하는 슬라이스만 학습한다.
        self.is_test = 'test' in data_dir
        self.transform = transform
        if (mode == 'kid'):
            self.data_dir = os.path.join(data_dir, 'kid_valid')
            vol_dir = os.path.join(self.data_dir, 'vol')
            self.lst_vol = sorted(list(glob(vol_dir + '\*\*.npy')))
            seg_dir = os.path.join(self.data_dir, 'seg')
            self.lst_seg = sorted(list(glob(seg_dir + '\*\*.npy')))

        elif (mode == 'tum'):
            self.data_dir = os.path.join(data_dir, 'tum_valid')
            vol_dir = os.path.join(self.data_dir, 'vol')
            self.lst_vol = sorted(list(glob(vol_dir + '\*\*.npy')))
            seg_dir = os.path.join(self.data_dir, 'seg')
            self.lst_seg = sorted(list(glob(seg_dir + '\*\*.npy')))

        else :
            self.data_dir = data_dir
            vol_dir = os.path.join(self.data_dir + 'vol')
            self.lst_vol = sorted(list(glob(vol_dir, '\*\*.npy')))
            seg_dir = os.path.join(self.data_dir + 'seg_kidney')
            self.lst_seg = sorted(list(glob(seg_dir, '\*\*.npy')))

    def __len__(self):
        return len(self.lst_vol)

    def __getitem__(self, idx):
        vol = np.load(self.lst_vol[idx])
        seg = np.load(self.lst_seg[idx])

        if vol.ndim == 2:
            vol = vol[:, :, np.newaxis]
        if seg.ndim == 2:
            seg = seg[:, :, np.newaxis]

        data = {'vol': vol, 'seg': seg}
        
        if self.transform: 
            data = self.transform(data)
        
        return data

# Transform 구현
class ToTensor(object):
    def __call__(self, data):
        vol, seg = data['vol'], data['seg']

        vol = vol.transpose((2, 0, 1)).astype(np.float64)
        seg = seg.transpose((2, 0, 1)).astype(np.uint8)

        data = {'vol': torch.from_numpy(vol), 'seg': torch.from_numpy(seg)}
        return data

class Normalization(object):
    def __call__(self, data):
        vol, seg = data['vol'], data['seg']

        vol -= np.min(vol)
        vol /= np.max(vol)
        data = {'vol': vol, 'seg': seg}
        return data

class RandomFlip(object):
    def __call__(self, data):
        vol, seg = data['vol'], data['seg']

        if np.random.rand() > 0.5:
            vol = np.fliplr(vol)
            seg = np.fliplr(seg)

        if np.random.rand() > 0.5:
            vol = np.flipud(vol)
            seg = np.flipud(seg)

        data = {'vol': vol, 'seg': seg}
        return data

transform1 = transforms.Compose([Normalization(), RandomFlip(), ToTensor()])
transform2 = transforms.Compose([Normalization(), ToTensor()])

"""
# check
data_train = kits19_Dataset(dir_train, transform=transform)
len = data_train.__len__()
print(len)
print(data_train.lst_vol[:10])
print(data_train.lst_seg[:10])

for i in range(10):
    idx = int(np.random.rand() * len)
    data = data_train.__getitem__(idx)
    print(idx)
    vol = data['vol']
    seg = data['seg']
    print(seg)
    print(type(seg))
    print(torch.min(seg))
    print(torch.max(seg))

    plt.subplot(121)
    plt.imshow(vol.squeeze())

    plt.subplot(122)
    plt.imshow(seg.squeeze())

    plt.show()
"""