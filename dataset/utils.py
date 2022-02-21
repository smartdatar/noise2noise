import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import random
import os
import argparse
import scipy.io as io
from tqdm import tqdm
import skimage

class AddGaussianNoise(object):
    """
    mean:均值
    variance：方差
    amplitude：幅值
    """

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = (variance/255)**2
        self.amplitude = amplitude

    def __call__(self, img):
        """
        :param img: type : array (h w c)
        :return: Image
        """

        # # img = np.array(img)
        # h, w, c = img.shape
        # N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        # N = np.repeat(N, c, axis=2)
        # img = N + img
        # img[img > 255] = 255  # 避免有值超过255而反转
        # img = Image.fromarray(img.astype('uint8')).convert('RGB')
        # return img

        img = skimage.util.random_noise(img, mode="gaussian", mean=self.mean, var=self.variance)
        img = np.uint8(img * 255)
        return Image.fromarray(img)


class AddSaltPepperNoise(object):
    def __init__(self, density=0, p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        """
        :param img: type: array (h, w, c)
        :return:  Image
        """
        if random.uniform(0, 1) < self.p:  # 概率的判断
            # img = np.array(img)  # 图片转numpy
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
            mask = np.repeat(mask, c, axis=2)  # 在通道的维度复制，生成彩色的mask
            img[mask == 0] = 0  # 椒
            img[mask == 1] = 255  # 盐
            img = Image.fromarray(img.astype('uint8')).convert('RGB')  # numpy转图片
            return img
        else:
            return img


class ExtractData:

    def __init__(self, args):
        self.args = args
        self.bsd_paths_train = ["dataset/data/BSD300/images/train"]
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop([256, 256]),
        ])
        # self.extract_bsd()

    def extract_bsd(self):
        img_list = []
        for path in self.bsd_paths_train:
            print("提取 %s" % path)

            for filename in tqdm(os.listdir(path), ncols=90):
                file_path = path + '/' + filename

                img = Image.open(file_path)
                for num_img in range(self.args.num_img):
                    croped_img = self.transform(img)

                    img_array = np.asarray(croped_img)
                    # img_array = img_array.transpose([2,0,1])
                    img_list.append(img_array)

        img_list = np.stack(img_list, axis=0)
        data = {'Img': img_list}
        path = "dataset/data/BSD300/BSD300_train.mat"
        print("正在保存 %s" % path)
        io.savemat(path, data)
        print("保存完成!!!!")


class PadImage:
    """
    调用__call__(img)，img格式为Image读取图片格式
    """
    def __init__(self):
        pass

    def __call__(self, img):
        """
        :param img: Image
        :return: Image
        """
        img = np.asarray(img).transpose([2, 0, 1])
        w = img.shape[2]
        h = img.shape[1]
        pw, ph = (w + 31) // 32 * 32 - w, (h + 31) // 32 * 32 - h
        padded_img = img
        if pw != 0 or ph != 0:
            padded_img = np.pad(img, ((0, 0), (0, ph), (0, pw)), 'reflect')
        padded_img = padded_img.transpose([1, 2, 0])
        return Image.fromarray(padded_img), w, h


class RecoverImage:
    """
    调用__call__(img)，输入img为tensor
    """
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        pass

    def __call__(self, img):
        """
        :param img: type: tensor (c, h, w)
        :return: type: tensor (c, h, w)
        """

        assert img.shape[0] == 3
        img = img[:, self.y:self.h, self.x:self.w]
        return img


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--extract', action="store_true", )
    parse.add_argument("--num_img", type=int, help="每张训练图像随机裁剪多少次", default=4)
    args = parse.parse_args()

    if args.extract:
        print("即将提取数据...")
        extract = ExtractData(args)
        extract.extract_bsd()
