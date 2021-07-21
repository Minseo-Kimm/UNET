from train_functions import *

import os
import numpy as np
import torch, gc
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

gc.collect()
torch.cuda.empty_cache()

# useSave를 True로 설정하면 저장된 네트워크를 불러와 학습을 시작한다.
# 1: lr = 1e-3, loss function = BCEWithLogitsLoss
# 2: output에 padding 적용 x, seg image를 388x388으로 crop함.
#    output 픽셀값을 0/1으로 바꾸어 validation 수행
# 3: unet의 마지막에 softmax를 추가함.
# 4: unet에 padding을 추가하여 output size가 512x512가 되도록 수정함.
#    메모리 문제로 batch size를 3으로 수정함.
#    learning rate를 1e-2로 수정함.
# 5: lr = 4e-2, loss function = MSELoss
#    average F1 score : 0.17
# 6: lr = 1e-2, loss function = BCELoss
#    data normalization에 clipping 추가 (-70, 310)
#    average F1 score : 0.16
# 7: lr = 2e-3, loss function = BCELoss
#    normalization 코드에 오류 발견하여 수정함.
version = 7
useSave = False         # 저장된 모델 사용하여 학습 시작

dataset_train = kits19_Dataset(dir_train, transform=transform1, mode=mode)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
num_train = len(dataset_train)
num_batch_train = np.ceil(num_train / batch_size)

dataset_val = kits19_Dataset(dir_val, transform=transform2, mode=mode)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
num_val = len(dataset_val)
num_batch_val = np.ceil(num_val / batch_size)

# 학습 전 저장된 네트워크가 있다면 불러오기
st_epoch = 0
pre_loss = 1
if (useSave):
    net, optim, st_epoch, pre_loss = load(ckpt_dir = ckpt_dir, net=net, optim=optim, pre_loss=pre_loss)

# Training
print("TRAINING STARTS")
for epoch in range(st_epoch, epochs):
    net.to(device)
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
        if (batch % 100 == 0) :
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | Loss %.4f" %
                (epoch + 1, epochs, batch, num_batch_train, loss.item())) 

        # Tensorboard 저장
        """
        seg = fn_tonumpy(seg)
        vol = fn_tonumpy(vol)
        output = fn_tonumpy(makePredict(output))

        writer_train.add_image('seg', seg, num_batch_train * (epoch) + batch, dataformats='NHWC')
        writer_train.add_image('vol', vol, num_batch_train * (epoch) + batch, dataformats='NHWC')
        writer_train.add_image('output', output, num_batch_train * (epoch) + batch, dataformats='NHWC')
        """
    
    loss_mean = np.mean(loss_arr)
    # writer_train.add_scalar('loss', loss_mean, (epoch + 1))

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            seg = data['seg'].to(device, dtype=torch.float)
            vol = data['vol'].to(device, dtype=torch.float)

            output = net(vol)

            # loss 계산
            loss = fn_loss(output, seg).item()
            loss_arr += [loss]
            #print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
            #      ((epoch + 1), epochs, batch, num_batch_val, loss))

            # Tensorboard 저장
            """
            writer_val.add_image('seg', seg, num_batch_train * (epoch) + batch, dataformats='NHWC')
            writer_val.add_image('vol', vol, num_batch_train * (epoch) + batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_batch_train * (epoch) + batch, dataformats='NHWC')
            """
    
    loss_mean = np.mean(loss_arr)
    print("-------------------------------------------------------------")
    print("EPOCH : %d | MEAN VALID LOSS : %.4f" % ((epoch + 1), loss_mean))
    print("-------------------------------------------------------------\n")
    # writer_val.add_scalar('loss', loss_mean, (epoch + 1))

    # loss가 최소이면 해당 네트워크를 저장
    if (loss_mean < pre_loss):
        save(ckpt_dir=ckpt_dir, net=net.cpu(), optim=optim, epoch=epoch + 1, ver=version, loss=loss_mean)
        pre_loss = loss_mean

"""
writer_train.close()
writer_val.close()
"""