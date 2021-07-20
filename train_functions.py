from kits19Loader import *
from unet import *

# Training parameters
lr = 4e-2
batch_size = 3
epochs = 8
mode = 'kid'            # 'all' = 모든 슬라이스, 'kid' = kidney 라벨링이 존재하는 슬라이스, 'tum' = tumor 라벨링이 존재하는 슬라이스
fn_loss = nn.MSELoss(reduction='mean').to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

# segmentation data를 output image의 크기에 맞게 crop
def cropimg(seg, output):
    s1, s2 = seg.size()[-2:]
    o1, o2 = output.size()[-2:]
    i1, i2 = (s1-o1)//2, (s2-o2)//2
    segcrop = seg[:, :, i1 : (i1+o1), i2 : (i2+o2)]
    return segcrop

# 네트워크 저장
def save(ckpt_dir, net, optim, epoch, ver, loss):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict(), 'loss': loss},
                "%s/ver%d_model_epoch%d.pth" % (ckpt_dir, ver, epoch))

# 네트워크 불러오기
def load(ckpt_dir, net, optim, pre_loss):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch, pre_loss
    
    ckpt_lst = os.listdir(ckpt_dir)
    #ckpt_lst.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    pre_loss = dict_model['loss']

    return net, optim, epoch, pre_loss

# output pixels의 값을 0또는 1으로 변환
def makePredict(output):
    result = torch.zeros_like(output)
    result[output > 0.5] = 1
    return result