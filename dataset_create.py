import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob

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

def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def saveImage(cids, dir, test=False):
    vol_dir = os.path.join(dir, 'vol')
    kid_dir = os.path.join(dir, 'seg_kidney')
    tum_dir = os.path.join(dir, 'seg_tumor')

    makeDir(vol_dir)

    for cid in cids:
        if (test) :
            vol = load_volume(cid)
            seg = None
        else :
            vol, seg = load_case(cid)

        vol_data = np.asarray(vol.dataobj, dtype=np.float32)
        slices, _, _ = vol_data.shape
        thisdir_vol = os.path.join(vol_dir, 'case_%03d' % cid)
        makeDir(thisdir_vol)
        for slice in range(slices):
            np.save(os.path.join(thisdir_vol, 'vol_slice_%05d' % slice), vol_data[slice, :, :])

        if seg is not None:     # test data일 경우에는 건너뜀
            makeDir(kid_dir)
            makeDir(tum_dir)
            seg_data = np.asarray(seg.dataobj, dtype=np.uint8)
            seg_kid = seg_data.copy()
            seg_kid[seg_data > 1] = 1   # 2로 라벨링되어있는 tumor 부분을 1로 바꿈
            seg_tum = np.zeros_like(seg_data)
            seg_tum[seg_data > 1] = 1   # tumor 부분만 1로 라벨링, 나머지는 0

            thisdir_kid = os.path.join(kid_dir, 'case_%03d' % cid)
            thisdir_tum = os.path.join(tum_dir, 'case_%03d' % cid)
            makeDir(thisdir_kid)
            makeDir(thisdir_tum)
            for slice in range(slices):
                np.save(os.path.join(thisdir_kid, 'kid_slice_%05d' % slice), seg_kid[slice, :, :])
                np.save(os.path.join(thisdir_tum, 'tum_slice_%05d' % slice), seg_tum[slice, :, :])
        print('%03d done' % cid)

def filterNone(cids, dir):    # dir 자리에는 train, val의 디렉토리가 들어감
    vol_dir = os.path.join(dir, 'vol')
    kid_dir = os.path.join(dir, 'seg_kidney')
    tum_dir = os.path.join(dir, 'seg_tumor')

    # ../train(or val)/kid_valid에는 kidney 라벨링이 존재하는 vol / kid 슬라이스만 저장
    # ../train(or val)/tum_valid에는 tumor 라벨링이 존재하는 vol / tum 슬라이스만 저장
    kid_valid_dir = os.path.join(dir, 'kid_valid')
    tum_valid_dir = os.path.join(dir, 'tum_valid')
    makeDir(kid_valid_dir)
    makeDir(tum_valid_dir)

    volkid_to = os.path.join(kid_valid_dir, 'vol')
    kid_to = os.path.join(kid_valid_dir, 'seg')
    voltum_to = os.path.join(tum_valid_dir, 'vol')
    tum_to = os.path.join(tum_valid_dir, 'seg')
    makeDir(volkid_to)
    makeDir(voltum_to)
    makeDir(kid_to)
    makeDir(tum_to)

    for cid in cids:
        vol_frompath = os.path.join(vol_dir, 'case_%03d' % cid)
        kid_frompath = os.path.join(kid_dir, 'case_%03d' % cid)
        tum_frompath = os.path.join(tum_dir, 'case_%03d' % cid)

        volkid_topath = os.path.join(volkid_to, 'case_%03d' % cid)
        voltum_topath = os.path.join(voltum_to, 'case_%03d' % cid)
        kid_topath = os.path.join(kid_to, 'case_%03d' % cid)
        tum_topath = os.path.join(tum_to, 'case_%03d' % cid)
        makeDir(volkid_topath)
        makeDir(voltum_topath)
        makeDir(kid_topath)
        makeDir(tum_topath)

        lst_vol = sorted(list(glob(vol_frompath + '\*.npy')))
        lst_kid = sorted(list(glob(kid_frompath + '\*.npy')))
        lst_tum = sorted(list(glob(tum_frompath + '\*.npy')))
        leng = len(lst_vol)

        for idx in range(leng):
            vol = np.load(lst_vol[idx])
            kid = np.load(lst_kid[idx])
            tum = np.load(lst_tum[idx])

            if (np.max(kid) != 0):
                np.save(os.path.join(volkid_topath, 'vol_slice_%05d' % idx), vol)
                np.save(os.path.join(kid_topath, 'kid_slice_%05d' % idx), kid)

            if (np.max(tum) != 0):
                np.save(os.path.join(voltum_topath, 'vol_slice_%05d' % idx), vol)
                np.save(os.path.join(tum_topath, 'kid_slice_%05d' % idx), kid)
                




# Dataset을 저장할 directory 생성
dir_data = 'E:\dataset'
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

makeDir(dir_save_train)
makeDir(dir_save_val)
makeDir(dir_save_test)

# train, val, test 폴더에 데이터를 numpy 형태로 저장
# train - 180개 (0 ~ 179)
# val - 30개 (180, 209) 
# test - 90개 (210, 299)

#saveImage(range(180), dir_save_train)
#saveImage(range(180, 210), dir_save_val)
#saveImage(range(210, 300), dir_save_test, test=True)

filterNone(range(180), dir_save_train)
filterNone(range(180, 210), dir_save_val)

"""
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