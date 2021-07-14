from kits19Loader import *
from unet import *

import os
import numpy as np
import torch, gc
from torch.utils.data import DataLoader
import torch.nn.functional as F

gc.collect()
torch.cuda.empty_cache()

# useSave를 True로 설정하면 저장된 네트워크를 불러와 학습을 시작한다.
# 2: output에 padding 적용 x, seg image를 388x388으로 crop함.
#    output 픽셀값을 0/1으로 바꾸어 validation 수행 
version = 2
useSave = False         # 저장된 모델 사용하여 학습 시작
mode = 'kid'            # 'all' = 모든 슬라이스, 'kid' = kidney 라벨링이 존재하는 슬라이스, 'tum' = tumor 라벨링이 존재하는 슬라이스
goTraining = True       # Training 수행
goValidation = True     # Validation 수행
goImaging = False       # 학습 결과를 이미지로 표현

dataset_train = kits19_Dataset(dir_train, transform=transform1, mode=mode)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
dataset_val = kits19_Dataset(dir_val, transform=transform2, mode=mode)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

# 데이터의 shape이 모두 512 x 512인지 확인 => case 160은 사이즈가 달라 제외하였다.
"""
ll = dataset_train.__len__()
data1 = dataset_train.__getitem__(0)
size = data1['vol'].shape

print(ll)
for i in range(ll):
    data = dataset_train.__getitem__(i)
    issize = data['vol'].shape
    if (size != issize) :
        print(issize)
        print("index is %d" % i)
        break
    elif (i % 10 == 0) : print(i)
"""

fn_loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

num_train = len(dataset_train)
num_val = len(dataset_val)

num_batch_train = np.ceil(num_train / batch_size)
num_batch_val = np.ceil(num_val / batch_size)

# segmentation data를 output image의 크기에 맞게 crop
def cropimg(seg, output):
    s1, s2 = seg.size()[-2:]
    o1, o2 = output.size()[-2:]
    i1, i2 = (s1-o1)//2, (s2-o2)//2
    segcrop = seg[:, :, i1 : (i1+o1), i2 : (i2+o2)]
    return segcrop

# output pixels의 값을 0또는 1으로 변환
def makePredict(output):
    soft = F.softmax(output, dim=0)
    result = torch.zeros_like(output)
    result[soft > 0.5] = 1
    return result

# 네트워크 저장
def save(ckpt_dir, net, optim, epoch, ver):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                "%s/ver%d_model_epoch%d.pth" % (ckpt_dir, ver, epoch))

# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

# 학습 전 저장된 네트워크가 있다면 불러오기
st_epoch = 0
if (useSave):
    net, optim, st_epoch = load(ckpt_dir = ckpt_dir, net=net, optim=optim)

if (goTraining) :
    # Training
    print("TRAINING STARTS")
    for epoch in range(st_epoch, epochs):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            seg = data['seg'].to(device=device, dtype=torch.float)
            vol = data['vol'].to(device=device, dtype=torch.float)

            output = net(vol)

            # backward pass
            optim.zero_grad()
            loss = fn_loss(output, cropimg(seg, output))
            loss.backward()

            optim.step()

            # loss 계산
            loss_arr += [loss.item()]
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | Loss %.4f" %
                (epoch + 1, epochs, batch, num_batch_train, np.mean(loss_arr)))
        
        # 네트워크를 중간중간 저장
        if epoch % 1 == 0:
            save(ckpt_dir=ckpt_dir, net=net.cpu(), optim=optim, epoch=epoch + 1, ver=version)

if (goValidation) :
    # Validation
    print("\n\nVALIDATION STARTS")

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            seg = data['seg'].to(device, dtype=torch.float)
            vol = data['vol'].to(device, dtype=torch.float)

            output = makePredict(net(vol))

            # loss 계산
            segcrop = cropimg(seg, output)
            loss = fn_loss(output, segcrop)
            loss_arr += [loss.item()]
            print("BATCH %04d / %04d | LOSS %.4f" % (batch, num_batch_val, np.mean(loss_arr)))

if (goImaging) :
    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            seg = data['seg'].to(device, dtype=torch.float)
            vol = data['vol'].to(device, dtype=torch.float)

            output = makePredict(net(vol))

            # loss 계산
            segcrop = cropimg(seg, output)
            loss = fn_loss(output, segcrop)
            loss_arr += [loss.item()]
            print("BATCH %04d / %04d | LOSS %.4f" % (batch, num_batch_val, np.mean(loss_arr)))
        
            # 이미지 확인
            for i in range(5):
                v = vol.cpu().numpy()[i, :, :, :]
                l = output.cpu().numpy()[i, :, :, :]
                r = segcrop.cpu().numpy()[i, :, :, :]
                plt.subplot(5, 3, (3*i+1))
                plt.imshow(v.squeeze())

                plt.subplot(5, 3, (3*i+2))
                plt.imshow(l.squeeze())

                plt.subplot(5, 3, (3*i+3))
                plt.imshow(r.squeeze())

            plt.show()
            break
