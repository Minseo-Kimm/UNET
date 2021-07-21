from train_functions import *

import os
import numpy as np
import torch, gc
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

gc.collect()
torch.cuda.empty_cache()

printmode = 'best'    # random or best
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

        output = makePredict(net(vol))

        # loss 계산
        loss = fn_loss(output, seg).item()
        loss_arr += [loss]
        accuracy = F1_score(output, seg)
        acc_arr += [accuracy]
        print("BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" % (batch, num_batch_val, loss, accuracy))

        if (accuracy > max_acc):
            max_acc = accuracy
            min_batch = batch
            min_data = data
        
        if (batch == rand_batch):
            rand_loss = loss
            rand_data = data

    if (printmode == 'best') :
        seg = min_data['seg'].to(device, dtype=torch.float)
        vol = min_data['vol'].to(device, dtype=torch.float)
        output = makePredict(net(vol))

        loss = fn_loss(output, seg).item()
        accuracy = F1_score(output, seg)
        print("PRINTED : BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" % (min_batch, num_batch_val, loss, accuracy))
        print("AVERAGE LOSS : %.4f | AVERAGE ACCURACY : %.4f" % (np.mean(loss_arr), np.mean(acc_arr)))
        for i in range(batch_size):
            v = vol.cpu().numpy()[i, :, :, :]
            l = output.cpu().numpy()[i, :, :, :]
            r = seg.cpu().numpy()[i, :, :, :]
            plt.subplot(4, 3, (3*i+1))
            plt.imshow(v.squeeze())

            plt.subplot(4, 3, (3*i+2))
            plt.imshow(l.squeeze())

            plt.subplot(4, 3, (3*i+3))
            plt.imshow(r.squeeze())
        plt.show()
    
    elif (printmode == 'random') :
        seg = rand_data['seg'].to(device, dtype=torch.float)
        vol = rand_data['vol'].to(device, dtype=torch.float)
        output = makePredict(net(vol))

        loss = fn_loss(output, seg).item()
        accuracy = F1_score(output, seg)
        print("PRINTED : BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" % (min_batch, num_batch_val, loss, accuracy))
        print("AVERAGE LOSS : %.4f | AVERAGE ACCURACY : %.4f" % (np.mean(loss_arr), np.mean(acc_arr)))
        for i in range(batch_size):
            v = vol.cpu().numpy()[i, :, :, :]
            l = output.cpu().numpy()[i, :, :, :]
            r = seg.cpu().numpy()[i, :, :, :]
            plt.subplot(4, 3, (3*i+1))
            plt.imshow(v.squeeze())

            plt.subplot(4, 3, (3*i+2))
            plt.imshow(l.squeeze())

            plt.subplot(4, 3, (3*i+3))
            plt.imshow(r.squeeze())
        plt.show()