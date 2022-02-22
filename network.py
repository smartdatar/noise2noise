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
        super(Conv,self).__init__()
        self.net = nn.Conv2d(in_features, out_features, kernel_size=kernel, padding=padding)
        _initWeight(self.net.weight, [kernel, kernel, in_features, out_features], gain)

    def forward(self, x):
        return self.net(x)


class ConvLR(nn.Module):
    def __init__(self, in_features, out_features, gain=np.sqrt(2), relu = True):
        super(ConvLR, self).__init__()
        lr = nn.LeakyReLU(0.1, inplace=True)
        if not relu:
            lr = nn.Identity()

        self.net = nn.Sequential(
            Conv(in_features, out_features, 3, 1, gain),
            nn.BatchNorm2d(out_features),
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



class UpSample(nn.Module):
    def __init__(self):
        pass


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self._block1 = nn.Sequential(                      # enc_conv1
            nn.Conv2d(3, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

        )
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),  # enc_conv2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1)

        )
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        )
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)

        )
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + 3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, img):
        pool1 = self._block1(img)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, img), dim=1)

        # Final activation
        return self._block6(concat1)