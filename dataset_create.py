import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from kits19.starter_code.utils import load_case, load_volume

"""
volume, segmentation = load_case(123)
print(volume.header.get_data_shape())           # (389, 512, 512)
print(segmentation.header.get_data_shape())     # (389, 512, 512)

print(volume.header.get_data_dtype())           # float64
print(segmentation.header.get_data_dtype())     # uint8

vol_data = volume.get_fdata()                   # numpy.darrary 형식으로 변경
seg_data = segmentation.get_fdata()             # numpy.darrary 형식으로 변경

print(vol_data.shape)
print(vol_data.dtype)
"""
# volume[0,:,:]

# Dataset을 저장할 directory 생성
dir_data = 'E:\dataset'
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)


# train, val, test 폴더에 데이터를 numpy 형태로 저장
# train - 180개 (0 ~ 179)
# val - 30개 (180, 209) 
# test - 90개 (210, 299)

for i in range(0, 180):
    vol, seg = load_case(i)
    vol_data = vol.get_fdata()
    seg_data = seg.get_fdata()
    slices, _, _ = vol_data.shape

    thispath = os.path.join(dir_save_train, 'case_%03d' % i)
    os.makedirs(thispath)
    for j in range (slices):
        np.save(os.path.join(thispath, 'vol_slice_%05d' % j), vol_data[j, :, :])
        np.save(os.path.join(thispath, 'seg_slice_%05d' % j), seg_data[j, :, :])
    print("done %03d" % i)

for i in range(180, 210):
    vol, seg = load_case(i)
    vol_data = vol.get_fdata()
    seg_data = seg.get_fdata()
    slices, _, _ = vol_data.shape

    thispath = os.path.join(dir_save_val, 'case_%03d' % i)
    os.makedirs(thispath)
    for j in range (slices):
        np.save(os.path.join(thispath, 'vol_slice_%05d' % j), vol_data[j, :, :])
        np.save(os.path.join(thispath, 'seg_slice_%05d' % j), seg_data[j, :, :])
    print("done %03d" % i)

for i in range(210, 300):
    vol = load_volume(i)
    vol_data = vol.get_fdata()
    slices, _, _ = vol_data.shape

    thispath = os.path.join(dir_save_test, 'case_%03d' % i)
    os.makedirs(thispath)
    for j in range (slices):
        np.save(os.path.join(thispath, 'vol_slice_%05d' % j), vol_data[j, :, :])
    print("done %03d" % i)

"""
vol, seg = load_case(123)
vol_data = vol.get_fdata()
seg_data = seg.get_fdata()
print(vol_data.shape)
print(vol_data[100, 250:260, 250:260])
print(seg_data[100, 250:260, 250:260])
# check
plt.subplot(121)
plt.imshow(vol_data[100, :, :], cmap='gray')
plt.title('volume')

plt.subplot(122)
plt.imshow(seg_data[100, :, :], cmap='gray')
plt.title('segmentation')
plt.show()
"""