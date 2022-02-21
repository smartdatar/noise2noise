from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import torch
import torch.nn as nn
import os
import numpy as np
from dataset.utils import PadImage, AddSaltPepperNoise, AddGaussianNoise


class Kodak(Dataset):
    def __init__(self, args):
        path = "./dataset/data/kodak/"
        self.img_paths = []
        self.add_noise = AddGaussianNoise(args.GaussianParams[0], args.GaussianParams[1])
        if args.noise == "Salt":
            self.add_noise = AddSaltPepperNoise(args.SaltParams[0], args.SaltParams[1])

        self.padImage = PadImage()
        self.totensor = transforms.ToTensor()
        for i in os.listdir(path):
            self.img_paths.append(path + i)

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        img, w, h = self.padImage(img)
        img = np.asarray(img)
        targets = img.copy().transpose([2, 0, 1])
        img = self.add_noise(img)
        img = self.totensor(img)

        return img, targets, int(self.img_paths[item][-6:-4]), w, h

    def __len__(self):
        return len(self.img_paths)


