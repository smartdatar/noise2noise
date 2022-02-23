from PIL import Image
from torch.utils.data import Dataset
import scipy.io as io
from torchvision.transforms import transforms
from pathlib import PureWindowsPath as wp
from dataset.utils import AddGaussianNoise, AddSaltPepperNoise
import numpy as np
import os
from dataset.utils import PadImage, AddSaltPepperNoise, AddGaussianNoise, AddPoissonNoise


class COCOTrain(Dataset):

    def __init__(self, args):
        self.method = args.method

        path = 'dataset/data/COCO_train.mat'

        self.data = io.loadmat(path)['Img']
        self.add_noise = AddGaussianNoise(args.GaussianParams[0], args.GaussianParams[1])
        if args.noise == "Salt":
            self.add_noise = AddSaltPepperNoise(args.SaltParams[0], args.SaltParams[1])
        elif args.noise == "Poisson":
            self.add_noise = AddPoissonNoise()

        self.transform = transforms.Compose([
            # add_noise,
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        input_img = self.data[index]
        target_img = input_img.copy()
        inputs = self.add_noise(input_img)
        if self.method == "n2n":
            targets = self.add_noise(target_img)
        elif self.method == "n2c":
            targets = target_img

        return self.transform(inputs), self.transform(targets)

    def __len__(self):
        return len(self.data)
