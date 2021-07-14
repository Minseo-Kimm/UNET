from kits19Loader import *
from unet import *

import os
import numpy as np
import torch, gc
from torch.utils.data import DataLoader

# useSave를 True로 설정하면 저장된 네트워크를 불러와 학습을 시작한다.
version = 1
useSave = True
mode = 'kid'    # 'all' = 모든 슬라이스, 'kid' = kidney 라벨링이 존재하는 슬라이스, 'tum' = tumor 라벨링이 존재하는 슬라이스
goTraining = False
goValidation = False
goImaging = True

dataset_train = kits19_Dataset(dir_train, transform=transform, mode=mode)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
dataset_val = kits19_Dataset(dir_val, transform=transform, mode=mode)
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

gc.collect()
torch.cuda.empty_cache()

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
            loss = fn_loss(output, seg)
            loss.backward()

            optim.step()

            # loss 계산
            loss_arr += [loss.item()]
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | Loss %.4f" %
                (epoch + 1, epochs, batch, num_batch_train, np.mean(loss_arr)))
        
        # 네트워크를 중간중간 저장
        if epoch % 1 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch + 1, ver=version)

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

            output = net(vol)

            # loss 계산
            loss = fn_loss(output, seg)
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

            output = net(vol)

            # loss 계산
            loss = fn_loss(output, seg)
            loss_arr += [loss.item()]
            print("BATCH %04d / %04d | LOSS %.4f" % (batch, num_batch_val, np.mean(loss_arr)))
        
            # 이미지 확인
            for i in range(5):
                v = vol.cpu().numpy()[i, :, :, :]
                l = output.cpu().numpy()[i, :, :, :]
                r = seg.cpu().numpy()[i, :, :, :]
                plt.subplot(5, 3, (3*i+1))
                plt.imshow(v.squeeze())

                plt.subplot(5, 3, (3*i+2))
                plt.imshow(l.squeeze())

                plt.subplot(5, 3, (3*i+3))
                plt.imshow(r.squeeze())

            plt.show()
            break
