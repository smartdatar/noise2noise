import torch.nn as nn
import numpy as np
import torch


def _initWeight(weight, shape, gain=np.sqrt(2)):
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)
    nn.init.normal_(weight, 0, std)


def upscale2d(x, factor=2):
    s = x.shape
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.repeat([1, 1, 1, factor, 1, factor])
    x = x.view(-1, s[1], s[2] * factor, s[3] * factor)
    return x


class Conv(nn.Module):
    def __init__(self, in_features, out_features, kernel=3, padding=1, gain=np.sqrt(2)):
        super(Conv, self).__init__()
        self.net = nn.Conv2d(in_features, out_features, kernel_size=kernel, padding=padding)
        _initWeight(self.net.weight, [kernel, kernel, in_features, out_features], gain)

    def forward(self, x):
        return self.net(x)


class ConvLR(nn.Module):
    def __init__(self, in_features, out_features, gain=np.sqrt(2), relu=True):
        super(ConvLR, self).__init__()
        lr = nn.LeakyReLU(0.1, inplace=True)
        if not relu:
            lr = nn.Identity()

        self.net = nn.Sequential(
            Conv(in_features, out_features, 3, 1, gain),
            # nn.BatchNorm2d(out_features),
            lr,
        )

    def forward(self, x):
        return self.net(x)


class MaxPool2d(nn.Module):
    def __init__(self, k=2):
        super(MaxPool2d, self).__init__()
        self.net = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.net(x)




class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.enc_block1 = nn.Sequential(
            ConvLR(3, 48),  # enc_conv0
            ConvLR(48, 48),  # enc_conv1
            MaxPool2d(2),  # MaxPool
        )
        self.enc_block2 = nn.Sequential(
            ConvLR(48, 48),  # enc_conv2
            MaxPool2d(2),
        )
        self.enc_block3 = nn.Sequential(
            ConvLR(48, 48),  # enc_conv3
            MaxPool2d(2),
        )
        self.enc_block4 = nn.Sequential(
            ConvLR(48, 48),  # enc_conv4
            MaxPool2d(2),
        )
        self.enc_block5 = nn.Sequential(
            ConvLR(48, 48),  # enc_conv5
            MaxPool2d(2),
            ConvLR(48, 48),  # enc_conv6

        )
        self.convt1 = nn.ConvTranspose2d(48, 48, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.dec_block1 = nn.Sequential(
            ConvLR(96, 96),  # dec_conv5
            ConvLR(96, 96),  # dec_conv5b
        )
        self.convt2 = nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_block2 = nn.Sequential(
            ConvLR(144, 96),  # dec_conv4
            ConvLR(96, 96),  # dec_conv4b
        )
        self.convt3 = nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.dec_block3 = nn.Sequential(
            ConvLR(144, 96),  # dec_conv3
            ConvLR(96, 96),  # dec_conv3b
        )
        self.convt4 = nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_block4 = nn.Sequential(
            ConvLR(144, 96),  # dec_conv2
            ConvLR(96, 96),  # dec_conv2b
        )
        self.convt5 = nn.ConvTranspose2d(96, 96, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_block5 = nn.Sequential(
            ConvLR(99, 64),  # dec_conv1a
            ConvLR(64, 32),  # dec_conv1b
        )

        self.conv = ConvLR(32, 3, relu=False, gain=1.0)


    def forward(self, img):


        skips = [img]    # channel : [3]
        x = self.enc_block1(img)
        skips.append(x)

        x = self.enc_block2(x)
        skips.append(x)  # channel : [3, 48]

        x = self.enc_block3(x)
        skips.append(x)   # channel : [3, 48, 48]

        x = self.enc_block4(x)
        skips.append(x)   # channel: [3, 48, 48, 48]

        x = self.enc_block5(x)

        x = self.convt1(x)

        x = torch.cat([x, skips.pop()], dim=1)

        x = self.dec_block1(x)
        x = self.convt2(x)
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec_block2(x)
        x = self.convt3(x)
        x = torch.cat([x, skips.pop()], dim=1)

        x = self.dec_block3(x)
        x = self.convt4(x)
        x = torch.cat([x, skips.pop()], dim=1)

        x = self.dec_block4(x)
        x = self.convt5(x)

        x = torch.cat([x, skips.pop()], dim=1)

        x = self.dec_block5(x)
        x = self.conv(x)
        return x



