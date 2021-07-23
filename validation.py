from train_functions import *

import os
import numpy as np
import torch, gc
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

gc.collect()
torch.cuda.empty_cache()

imgdir = 'C:/Users/msKim/Desktop/Unet/result/ver%d/' % version 
dataset_val = kits19_Dataset(dir_val, transform=transform2, mode=mode)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
num_val = len(dataset_val)
num_batch_val = np.ceil(num_val / batch_size)

# 저장된 네트워크 불러오기
net, _, _, _ = load(ckpt_dir = ckpt_dir, net=net, optim=optim, pre_loss=1)

# Validation
print("\nVALIDATION STARTS")

with torch.no_grad():
    net.to(device)
    net.eval()
    loss_arr = []
    acc_arr = []
    max_acc = 0
    rand_batch = int(np.random.rand() * num_batch_val)

    for batch, data in enumerate(loader_val, 1):
        # forward pass
        seg = data['seg'].to(device, dtype=torch.float)
        vol = data['vol'].to(device, dtype=torch.float)
        output = net(vol)
        output2 = makePredict(output)

        # loss 계산
        loss = fn_loss(output, seg).detach().item()
        loss_arr += [loss]
        accuracy = F1_score(output2, seg)
        acc_arr += [accuracy]
        print("BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" % (batch, num_batch_val, loss, accuracy))

        # 이미지 저장
        if not (batch == num_batch_val) :
            plot = plt.figure()
            for i in range(batch_size):
                input = vol.cpu().numpy()[i, :, :, :]
                result = output2.cpu().numpy()[i, :, :, :]
                label = seg.cpu().numpy()[i, :, :, :]
                plt.subplot(batch_size, 3, (batch_size*i+1))
                plt.imshow(input.squeeze())

                plt.subplot(batch_size, 3, (batch_size*i+2))
                plt.imshow(result.squeeze())

                plt.subplot(batch_size, 3, (batch_size*i+3))
                plt.imshow(label.squeeze())
            title = "BATCH %03d | SCORE %.4f" % (batch, accuracy)
            plt.suptitle(title)
            plt.savefig(imgdir + '%03d_%.4f.png' % (batch, accuracy))
            plt.close()

        if (accuracy > max_acc):
            max_acc = accuracy
            max_batch = batch
            max_loss = loss
            max_data = data
        
        if (batch == rand_batch):
            rand_acc = accuracy
            rand_loss = loss
            rand_data = data

    # print best case
    plot1 = plt.figure(1)
    max_seg = max_data['seg'].to(device, dtype=torch.float)
    max_vol = max_data['vol'].to(device, dtype=torch.float)
    max_output = net(max_vol)
    max_output2 = makePredict(max_output)

    print("PRINTED : BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" % (max_batch, num_batch_val, max_loss, max_acc))
    for i in range(batch_size):
        v = max_vol.cpu().numpy()[i, :, :, :]
        # out1 = max_output.cpu().numpy()[i, :, :, :]
        out2 = max_output2.cpu().numpy()[i, :, :, :]
        r = max_seg.cpu().numpy()[i, :, :, :]
        plt.subplot(batch_size, 3, (batch_size*i+1))
        plt.imshow(v.squeeze())

        plt.subplot(batch_size, 3, (batch_size*i+2))
        plt.imshow(out2.squeeze())

        plt.subplot(batch_size, 3, (batch_size*i+3))
        plt.imshow(r.squeeze())
    max_title = "BEST: BATCH %04d | SCORE %.4f" % (max_batch, max_acc)
    plt.suptitle(max_title)

    # print random case
    plot2 = plt.figure(2)
    rand_seg = rand_data['seg'].to(device, dtype=torch.float)
    rand_vol = rand_data['vol'].to(device, dtype=torch.float)
    rand_output = net(rand_vol)
    rand_output2 = makePredict(rand_output)

    print("PRINTED : BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" % (rand_batch, num_batch_val, rand_loss, rand_acc))
    print("AVERAGE LOSS : %.4f | AVERAGE ACCURACY : %.4f" % (np.mean(loss_arr), np.mean(acc_arr)))
    for i in range(batch_size):
        v = rand_vol.cpu().numpy()[i, :, :, :]
        # out1 = rand_output.cpu().numpy()[i, :, :, :]
        out2 = rand_output2.cpu().numpy()[i, :, :, :]
        r = rand_seg.cpu().numpy()[i, :, :, :]
        plt.subplot(batch_size, 3, (batch_size*i+1))
        plt.imshow(v.squeeze())

        plt.subplot(batch_size, 3, (batch_size*i+2))
        plt.imshow(out2.squeeze())

        plt.subplot(batch_size, 3, (batch_size*i+3))
        plt.imshow(r.squeeze())
    rand_title = "RANDOM: BATCH %04d | SCORE %.4f" % (rand_batch, rand_acc)
    plt.suptitle(rand_title)
    plt.show()