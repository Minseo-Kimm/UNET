from train_functions import *

import os
import numpy as np
import torch, gc
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))
memo = open(os.path.join(log_dir, 'ver%d.txt' % version), 'a')

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
#    average F1 score : 0.17 (epoch 7)
# 6: lr = 1e-2, loss function = BCELoss
#    data normalization에 clipping 추가 (-70, 310)
#    average F1 score : 0.16 (epoch 10)
# 7: lr = 2e-3, loss function = BCELoss
#    normalization 코드에 오류 발견하여 수정함.
#    average F1 score : 0.19 (epoch 4)
# 8: lr = 1e-4
#    unet의 마지막 softmax 전 단계에 tanh 추가
#    validation에서 픽셀값이 0.3을 넘으면 1으로 인식하도록 수정
# 9: unet의 마지막 softmax를 tanh + ReLU로 변경
#    픽셀값 변환 역치를 0.4로 수정
#    average F1 score : 0.86 (epoch 8)
# 10: lr = 1e-5, 픽셀값 변환 역치 0.5로 수정
#     average F1 score : 0.9046 (epoch 29)
# 11: output layer 3으로 수정
#     loss func: CrossEntropyLoss
#     average dice score : 0.6584 (epoch 15)
# 12: unet의 마지막 tanh + ReLU를 softmax로 변경
#     lr = 1e-4, 픽셀값 변환 역치 0.4
#     average dice score : 0.6632 (epoch 30)
# 13: output layer 2로 수정 (background / kidney(include tumor)만 분류)
#     average dice score : 0.9222 (epoch 36)

useSave = False         # 저장된 모델 사용하여 학습 시작

torch.manual_seed(300)

dataset_train = kits19_Dataset(dir_train, transform=transform1, mode=mode)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
num_train = len(dataset_train)
num_batch_train = np.ceil(num_train / batch_size)

dataset_val = kits19_Dataset(dir_val, transform=transform2, mode=mode)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=True)
num_val = len(dataset_val)
num_batch_val = np.ceil(num_val / batch_size)

# 학습 전 저장된 네트워크가 있다면 불러오기
st_epoch = 0
pre_loss = 1
pre_score = 0
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
        seg = seg.long().squeeze()
        loss = fn_loss(output, seg) # output shape: [3, 3, 512, 512], seg shape: [3, 512, 512]
        loss.backward()
        optim.step()

        # loss 계산
        loss_arr += [loss.item()]
        if (batch % 100 == 0) :
            num = (batch //100) % 10
            print(num, end='')
        if (batch % 500 == 1) :
            res = "TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | Loss %.4f\n" % (epoch + 1, epochs, batch, num_batch_train, np.mean(loss_arr))
            memo.write(res)

    with torch.no_grad():
        print("\nVALIDATION STARTS")
        net.eval()
        loss_arr = []
        acc_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            seg = data['seg'].to(device, dtype=torch.float)
            vol = data['vol'].to(device, dtype=torch.float)

            output = net(vol)

            # loss 계산
            loss = fn_loss(output, seg.long().squeeze()).detach().item()
            acc = Dice_score(makePredict(output), seg, mode)
            loss_arr += [loss]
            acc_arr += [acc]
            #print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
            #      ((epoch + 1), epochs, batch, num_batch_val, loss))
    
    loss_mean = np.mean(loss_arr)
    acc_mean = np.mean(acc_arr)
    print("-------------------------------------------------------------")
    res = "EPOCH : %d | MEAN VALID LOSS : %.4f | SCORE : %.4f\n" % ((epoch + 1), loss_mean, acc_mean)
    memo.write(res)
    print("EPOCH : %d | MEAN VALID LOSS : %.4f | SCORE : %.4f" % ((epoch + 1), loss_mean, acc_mean))
    print("-------------------------------------------------------------\n")

    # score가 최소이면 해당 네트워크를 저장
    # from ver9: 모든 네트워크 저장
    if (True):
        save(ckpt_dir=ckpt_dir, net=net.cpu(), optim=optim, epoch=epoch + 1, ver=version, loss=loss_mean)
        pre_score = acc_mean

memo.close()