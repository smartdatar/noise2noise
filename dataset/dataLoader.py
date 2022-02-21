import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.BSD300 import BSDDataTrain, BSDDataTest
from dataset.Kodak import Kodak

class Data:
    def __init__(self, args):
        if args.dataset == "BSD300":
            data = BSDDataTrain(args)

        self.dataLoader = DataLoader(data, shuffle=True, batch_size=args.batch_size)
        self.len = data.__len__()
        varityData = Kodak(args)
        if args.varitydata == "BSD300":
            varityData = BSDDataTest(args)
        self.varityLoader = DataLoader(varityData, shuffle=True, batch_size=1)


