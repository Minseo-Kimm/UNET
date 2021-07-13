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
    def __init__(self, data_dir, transform=None):   # data_dir 자리에는 train | val | test가 온다
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = 'test' in data_dir

        lst_data = sorted(list(glob(self.data_dir + '\*\*.npy')))

        lst_vol = []
        lst_seg = []
        for f in lst_data:
            if ('vol' in f) : lst_vol.append(f)
            elif ('seg' in f) : lst_seg.append(f)

        self.lst_vol = lst_vol
        self.lst_seg = lst_seg

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

transform = transforms.Compose([Normalization(), RandomFlip(), ToTensor()])

"""
# check
data_train = kits19_Dataset(dir_train, transform=transform)
print(data_train.__len__())
print(data_train.lst_vol[:10])
print(data_train.lst_seg[:10])

data = data_train.__getitem__(1000)
vol = data['vol']
seg = data['seg']
print(vol.shape)
print(type(vol))

plt.subplot(121)
plt.imshow(vol.squeeze())

plt.subplot(122)
plt.imshow(seg.squeeze())

plt.show()
"""