import os

import torch
import numpy as np
from data_loader import DeepFakeDataset
import time
from torchvision.transforms import transforms
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from double_unet import Double_UNet
from ae_models import U_Net, AttU_Net, NestedUNet, Unet_dict, R2AttU_Net, R2U_Net


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(scale),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class OutConvAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvAE, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3),
                                  nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=2),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, model='big'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if model == 'big':
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            self.up1 = Up(1024, 512 // factor, bilinear)
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            self.up4 = Up(128, 64, bilinear)
            self.outc = OutConv(64, n_classes)
        elif model == 'small':
            self.inc = DoubleConv(n_channels, 16)
            self.down1 = Down(16, 32)
            self.down2 = Down(32, 64)
            self.down3 = Down(64, 128)
            factor = 2 if bilinear else 1
            self.down4 = Down(128, 256 // factor)
            self.up1 = Up(256, 128 // factor, bilinear)
            self.up2 = Up(128, 64 // factor, bilinear)
            self.up3 = Up(64, 32 // factor, bilinear)
            self.up4 = Up(32, 16, bilinear)
            self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


class UpAE(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale=2, padding=1, kernel_size=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, padding)

    def forward(self, x1):
        x1 = self.up(x1)
        x1 = self.conv(x1)

        return x1



class AE1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = UpAE(1024, 512 // factor, bilinear)
        self.up2 = UpAE(512, 256 // factor, bilinear, padding=3)
        self.up3 = UpAE(256, 128 // factor, bilinear, padding=2)
        self.up4 = UpAE(128, 64, bilinear, padding=3)
        self.outc = OutConvAE(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        logits = logits[:, :, 0:-1, 0:-1]

        return logits


class AE2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 1024 // factor)
        self.up1 = UpAE(1024, 512 // factor, bilinear)
        self.up2 = UpAE(512, 256 // factor, bilinear)
        self.up4 = UpAE(256, 64, bilinear)
        self.outc = OutConvAE(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x5 = self.down4(x3)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up4(x)
        logits = self.outc(x)
        logits = logits[:, :, 0:-9, 0:-9]

        return logits




if __name__ == "__main__":

    dataset_adr = r'D:\saved_img'  #r'F:\ff++\saved_images'  # r'E:\saved_img'
    train_file_path = r'train_test_combined_final.xlsx'
    img_type = 'fullface'

    dataset = 'FF'
    model_type = 'AE_U_Net_l1_'
    ######################
    lr = 1e-4
    #####################3
    weight_decay = 0
    nr_epochs = 15
    lr_decay = 0.9
    test_data_frequency = 1
    train_batch_size = 8
    test_batch_size = 1024
    gradient_clipping_value = None  # 1
    model_param_adr = None #r'D:\saved_model\AE_Unet_dict_l1__fullface_epoch_2_param_all_23_642.pkl'    # None if new training

    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(20, 25)),
        transforms.ColorJitter(hue=0.05, saturation=0.05, brightness=0.2),
        transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
        transforms.RandomAdjustSharpness(2, p=1),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Resnet and VGG19 expects to have data normalized this way (because pretrained)
    ])

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    denorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
        # always_apply=True,
        # max_pixel_value=1.0
    )

    data_train = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf,
                                 batch_size=6, train=True, image_type=img_type, dataset=dataset, frames=6)

    #model = UNet(n_channels=3, n_classes=3, bilinear=True, model='small')

    # from ae_models import U_Net, AttU_Net, NestedUNet, Unet_dict, R2AttU_Net, R2U_Net
    #
    # #https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets
    # models = []
    # model = U_Net()
    # models.append(model)
    # model = R2U_Net()
    # models.append(model)
    # model = AttU_Net()
    # models.append(model)
    # model= R2AttU_Net()
    # models.append(model)
    # model = NestedUNet()
    # models.append(model)
    # model = Unet_dict(n_labels=3)
    # models.append(model)
    #
    # losses = [torch.nn.L1Loss().to('cuda'), torch.nn.MSELoss().to('cuda')]
    #
    # first = 0
    # skip = 2
    # for loss in losses:
    #     for model in models:
    #
    #         if first < skip:
    #             first += 1
    #             continue
    #
    #         model_type = 'AE_' + model._get_name() + '_l1_'
    #         print('MODEL', model_type)
    #         print('LOSS', loss)
    #
    #         model.to('cuda')
    #         #loss = torch.nn.MSELoss().to('cuda')
    #         #loss = torch.nn.L1Loss().to('cuda')
    #
    #         optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #
    #         epoch_done = 0
    #         if model_param_adr:
    #             model.load_state_dict(torch.load(model_param_adr))
    #
    #         print('Model loaded')
    #
    #         for epoch in range(1, 3):
    #             losses = 0
    #             for i in range(len(data_train)//4):
    #
    #                 data_train.shuffle()
    #                 t = data_train[i][0]  # [0, :, :, :, :]
    #                 t = t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3], t.shape[4])
    #                 t = t.to('cuda')
    #                 print(f'{i}/{len(data_train)}')
    #
    #                 for j in range(128):
    #
    #                     # t = data_train[i][0]  #[0, :, :, :, :]
    #                     # t = t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3], t.shape[4])
    #                     # t = t.to('cuda')
    #                     data_in = t[j*8:(1+j)*8]
    #                     x = model(data_in)
    #
    #                     # Limits
    #                     x[x > 1] = 1
    #                     x[x < -1] = -1
    #
    #                     l = loss(x, data_in)
    #
    #                     optimizer.zero_grad()
    #                     l.backward()
    #                     optimizer.step()
    #
    #                     losses += l
    #
    #                 print('Loss', losses/i/128)
    #
    #             print(f'\n\n\n\n Epoch {epoch + 1}')
    #
    #             # Saving model
    #             torch.save(model.state_dict(),
    #                        os.path.join(r'D:\saved_model', model_type + '_' + img_type + '_epoch_' + str(epoch) + '_param_' + dataset + '_' +
    #                                     str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4]) + '.pkl'))
    #
    #         print('yo')
    #
    #


    # Evaluation
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    denorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
        # always_apply=True,
        # max_pixel_value=1.0
    )

    # model_param_adr = r'D:\saved_model\AE_unet_fullface_epoch_4_param_FF++_1310_2315.pkl'    # None if new training
    # model = AE1(n_channels=3, n_classes=3)
    # #model = UNet(n_channels=3, n_classes=3, bilinear=True, model='small')
    # #model = Unet_dict(n_labels=3)
    # model.to('cuda')
    device = 'cuda'
    models=[]

    model_redo = Unet_dict(n_labels=3)
    model_redo.to(device)
    model_redo.load_state_dict(torch.load(r'D:\saved_model\AE_Unet_dict_l1__fullface_epoch_1_param_all_23_635.pkl'))
    models.append(model_redo)

    model = model_redo
    if model_param_adr:
        model.load_state_dict(torch.load(model_param_adr))



    data_test = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf,
                                 batch_size=32, train=False, image_type=img_type, dataset=dataset, frames=2)
    t = data_test[10][0]
    t = t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3], t.shape[4])
    t = t.to('cuda')
    with torch.no_grad():
        x = model(t)


    t_init = denorm(t)
    t_init = t_init.cpu().detach().numpy()
    t_init[t_init < 0] = 0
    t_init[t_init > 1] = 1
    t_init = np.uint8(t_init * 255)
    rez = denorm(x)
    rez = rez.cpu().detach().numpy()
    rez[rez < 0] = 0
    rez[rez > 1] = 1
    rez = np.uint8(rez * 255)

    print('Mean', (t-x).abs().mean())
    print('Mean no abs t - x', (t - x).mean())

    fignr = 6
    print(data_test[0][1][fignr//8])
    print('Mean img', np.abs(t_init[fignr].astype('float') - rez[fignr].astype('float')).mean())

    plt.figure(1)
    plt.imshow(t_init[fignr].transpose(1, 2, 0))
    plt.title('Init')
    plt.figure(2)
    plt.imshow(rez[fignr].transpose(1, 2, 0))
    plt.title('Autoencoder')
    plt.show()

    from skimage.color import rgb2hsv, rgb2gray, rgb2yuv

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from skimage.io import imread, imshow

    # from skimage import color, exposure, transform
    # from skimage.exposure import equalize_hist
    #
    # dark_image_grey1 = rgb2gray(t_init.mean(0).transpose(1, 2, 0))
    #
    #
    # dark_image_grey_fourier1 = np.fft.fftshift(np.fft.fft2(dark_image_grey1))
    # plt.figure(num=3, figsize=(8, 6), dpi=80)
    # plt.imshow(np.log(abs(dark_image_grey_fourier1)), cmap='gray')
    #
    #
    # #######
    # dark_image_grey = rgb2gray(rez.mean(0).transpose(1, 2, 0))
    #
    # dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(dark_image_grey))
    # plt.figure(num=4, figsize=(8, 6), dpi=80)
    # plt.imshow(np.log(abs(dark_image_grey_fourier)), cmap='gray')
    #
    # dark_image_grey_fourier2 = np.abs(dark_image_grey_fourier1 - dark_image_grey_fourier)
    # plt.figure(num=5, figsize=(8, 6), dpi=80)
    # plt.imshow(np.log(abs(dark_image_grey_fourier2)), cmap='gray')
    #
    # plt.show()

    # from scipy.fftpack import dct, idct
    #
    #
    # def dct2(a):
    #     return dct(dct(a.T, norm='ortho').T, norm='ortho')
    #
    #
    #
    #
    # vect = []
    # print(len(data_train))
    # for j in range(len(data_train)):
    #     print(j)
    #     t = data_train[j][0]
    #     t = t[data_train[j][1] == 1]
    #     t = t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3], t.shape[4])
    #     t= t.to('cuda')
    #
    #     with torch.no_grad():
    #         x = model(t)
    #
    #     rez = denorm(x)
    #     rez = rez.cpu().detach().numpy()
    #     rez[rez < 0] = 0
    #     rez[rez > 1] = 1
    #     rez = np.uint8(rez * 255)
    #
    #     # t_init = denorm(t)
    #     # t_init[t_init < 0] = 0
    #     # t_init[t_init > 1] = 1
    #     # t_init = np.uint8(t_init * 255)
    #     for i in range(len(t)):
    #         di2 = rgb2gray(rez[i].transpose(1, 2, 0))
    #         imF = dct2(di2)
    #         vect.append(imF)
    #
    # imF = np.mean(vect, 0)
    #
    # plt.figure(num=5, figsize=(8, 6), dpi=80)
    # plt.imshow(np.log(abs(imF)))
    # plt.title(f'CDF for {dataset}')
    # plt.show()

    print('yo')
