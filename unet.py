import torch
import torch.nn as nn

# Training parameters
lr = 1e-3
batch_size = 5
epochs = 10


ckpt_dir = 'C:/Users/msKim/Desktop/Unet/ckpt'   # train된 네트워크가 저장될 checkpoint dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## DataLoader 구현하기
## Data Augmentation

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_chs, out_chs, kernel_size=3, stride=1, padding=0, bias=True):
            # ConvNet, batchnorm, ReLU를 차례로 수행 (그림의 파란 화살표)
            layers = []
            layers += [nn.Conv2d(in_channels=in_chs, out_channels=out_chs, 
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_chs)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)
            return cbr
        
        # Contracting Path
        self.enc1_1 = CBR2d(in_chs=1, out_chs=64)
        self.enc1_2 = CBR2d(in_chs=64, out_chs=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_chs=64, out_chs=128)
        self.enc2_2 = CBR2d(in_chs=128, out_chs=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_chs=128, out_chs=256)
        self.enc3_2 = CBR2d(in_chs=256, out_chs=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.enc4_1 = CBR2d(in_chs=256, out_chs=512)
        self.enc4_2 = CBR2d(in_chs=512, out_chs=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_chs=512, out_chs=1024)

        # Expanding Path
        self.dec5_1 = CBR2d(in_chs=1024, out_chs=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4_2 = CBR2d(in_chs=1024, out_chs=512)
        self.dec4_1 = CBR2d(in_chs=512, out_chs=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3_2 = CBR2d(in_chs=512, out_chs=256)
        self.dec3_1 = CBR2d(in_chs=256, out_chs=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_2 = CBR2d(in_chs=256, out_chs=128)
        self.dec2_1 = CBR2d(in_chs=128, out_chs=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_2 = CBR2d(in_chs=128, out_chs=64)
        self.dec1_1 = CBR2d(in_chs=64, out_chs=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1,
                            kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):   # x는 input image

        def tensorcat(enc, unpool):
            [_, _, _, n1] = enc.size()
            [_, _, _, n2] = unpool.size()
            st = (n1 - n2) // 2
            cropped = enc[:, :, st : st+n2, st : st+n2]
            cat = torch.cat((cropped, unpool), dim=1)   # dim = [0: batch, 1: channel, 2: height, 3: width]
            return cat

        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = tensorcat(enc4_2, unpool4)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = tensorcat(enc3_2, unpool3)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = tensorcat(enc2_2, unpool2)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = tensorcat(enc1_2, unpool1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        fc = self.fc(dec1_1)
        pad = nn.ReflectionPad2d(94)
        fc = pad(fc)
        return fc


net = UNet().to(device)