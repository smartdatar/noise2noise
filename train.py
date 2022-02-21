import torch
from dataset.dataLoader import Data
from torchvision.transforms import transforms
from dataset.utils import RecoverImage
from PIL import Image
from network import AutoEncoder
from torch.optim import Adam, SGD
import torch.nn as nn
import os
import shutil
import time
import skimage
import numpy as np
import logging

class Trainer:
    """
    This is a class for training data, call this class
    to start the training journey.
    """

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self._init_net()
        self._load_data()
        self.start_time = time.time()
        self.sum_cost_time = 0
        if not os.path.exists('result'):
            os.mkdir('result')
        dirlist = os.listdir('result')
        save_file = "%05d-autoencoder-%s-%s/" % (len(dirlist) + 1,
                                                 self.args.dataset,
                                                 self.args.method,)
        self.save_path = "result/" + save_file
        os.mkdir(self.save_path)
        # self.run()

        self.logger = args.logger
        self.setLog()

    def setLog(self):

        self.psnr_file = open(self.save_path+'psnr.csv','a')
        self.psnr_file.write("iter,psnr")
        self.psnr_file.flush()

        fileHandler = logging.FileHandler(self.save_path+"log.txt")
        fileHandler.setLevel(logging.INFO)
        fileFormatter = logging.Formatter("%(message)s - %(asctime)s")
        fileHandler.setFormatter(fileFormatter)
        self.logger.addHandler(fileHandler)
        self.logger.info("start run...")
        self.logger.info('------------------    args   --------------------')
        for k in list(vars(self.args).keys()):
            self.logger.info('%20s: %-35s' % (k, vars(self.args)[k]))
        self.logger.info('-------------------   args   ------------------\n\n')

    def _init_net(self):
        self.model = AutoEncoder()
        if self.args.varity:
            self.model.load_state_dict(torch.load(self.args.varitymodel,map_location="cpu"))

        self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # self.optimizer = SGD(self.model.parameters(), lr=1e-3)
        if self.args.loss == "L2":
            self.criterion = nn.MSELoss()
        elif self.args.loss == "L1":
            self.criterion = nn.L1Loss()
        self.losses = []

    def _load_data(self):
        data = Data(self.args)
        self.dataLoader = data.dataLoader
        self.len = data.len

        self.varityLoader = data.varityLoader

    def run(self):
        for epoch in range(1, self.args.epochs + 1):
            mean_loss = []
            for batch, (inputs, targets) in enumerate(self.dataLoader, 1):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                output = self.model(inputs)
                loss = self.criterion(output, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mean_loss.append(loss.item())
            self.losses.append(sum(mean_loss)/len(mean_loss))
            if epoch % self.args.eval_interval == 0:
                self.verify(epoch)

        torch.save(self.model.state_dict(), self.save_path + 'model.pt')
        loss_array = np.array(self.losses)
        np.savetxt(self.save_path+'loss.txt', loss_array)
        self.logger.info("successful!!!")
        self.logger.info(
            "总耗时: %dh%dm%ds" % (self.sum_cost_time // 3600, (self.sum_cost_time % 3600) // 60, self.sum_cost_time % 60))

    def save_img(self, ori_img, noise_img, pred_img, code, w, h, epoch):


        transform = transforms.Compose([
            RecoverImage(0, 0, w, h),
            transforms.ToPILImage(),
        ])
        recover_ori_img = transform(ori_img)
        recover_noise_img = transform(noise_img)
        recover_pred_img = transform(pred_img)
        if epoch == self.args.eval_interval:
            recover_ori_img.save(self.save_path + "img_%d_ori.png" % (code,))
        recover_noise_img.save(self.save_path + "epoch_%d_img_%d_noise.png" % (epoch, code,))
        recover_pred_img.save(self.save_path + "epoch_%d_img_%d_pred.png" % (epoch, code,))

    def verify(self, epoch):
        psnrs = []
        for img, targets, code, w, h in self.varityLoader:

            noise_img = img.clone().detach()
            img = img.to(self.device)
            output = self.model(img)
            self.save_img(targets[0], noise_img[0], output[0].cpu().detach(), code[0].item(), w[0].item(), h[0].item(), epoch)
            ori_img = targets[0].numpy().transpose([1, 2, 0])
            pred_img = output[0].detach().cpu().numpy().transpose([1, 2, 0])
            pred_img = np.uint8(pred_img*255)
            pred_img = np.clip(pred_img, 0, 255)

            psnr = skimage.metrics.peak_signal_noise_ratio(ori_img, pred_img)
            psnrs.append(psnr)

        b = time.time()
        mean_psnr = sum(psnrs)/len(psnrs)
        self.psnr_file.write("\n%d,%.3f"%(epoch, mean_psnr))
        self.psnr_file.flush()
        used_time = int(b - self.start_time)
        used_time_str = "%dm%02ds" % (used_time // 60, used_time % 60)
        self.logger.info("iter: %d/%d [=======]  used_time:  %s  mean_psnr: %.4f      " % (epoch, self.args.epochs, used_time_str, mean_psnr))
        self.start_time = b
        self.sum_cost_time += used_time
