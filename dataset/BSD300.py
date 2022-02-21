from PIL import Image
from torch.utils.data import Dataset
import scipy.io as io
from torchvision.transforms import transforms
from dataset.utils import AddGaussianNoise, AddSaltPepperNoise
import numpy as np
import os
from dataset.utils import PadImage, AddSaltPepperNoise, AddGaussianNoise

class BSDDataTrain(Dataset):

    def __init__(self, args):
        self.method = args.method

        path = 'dataset/data/BSD300/BSD300_train.mat'

        self.data = io.loadmat(path)['Img']
        self.add_noise = AddGaussianNoise(args.GaussianParams[0], args.GaussianParams[1])
        if args.noise == "Salt":
            self.add_noise = AddSaltPepperNoise(args.SaltParams[0], args.SaltParams[1])
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


class BSDDataTest(Dataset):
    def __init__(self, args):
        path = "./dataset/data/BSD300/images/test/"
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