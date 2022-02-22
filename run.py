"""

"""
import argparse
from train import Trainer
import torch
import logging
parse = argparse.ArgumentParser()
parse.add_argument('--epochs', type=int, help="迭代次数", default=3600)
parse.add_argument('--method', type=str, choices=['n2n', 'n2c'], help="方法", default="n2n")
parse.add_argument('--batch_size', type=int, help="批次大小", default=30)
parse.add_argument("--lr", type=float, help="学习率", default=0.9)
parse.add_argument('--weight_decay', type=float, help="权重衰减参数", default=0)
parse.add_argument('--eval_interval', type=int, help="迭代多少次进行一次验证", default=30)
parse.add_argument("--varity", action="store_true", help="是否直接验证，默认是false")
parse.add_argument("--loss", type=str, choices=["L1", "L2"], default="L2")
parse.add_argument('--dataset', type=str, choices=['BSD300', ], default='BSD300')
parse.add_argument("--varitydata", type=str, choices=["BSD300", "kodak"], default="kodak")
parse.add_argument("--varitymodel", type=str, help="验证模型的参数文件所在路径")
parse.add_argument('--noise', type=str, choices=['Gaussian', 'Salt'], default="Gaussian")
parse.add_argument('--GaussianParams', type=float, nargs='+', default=[0., 25.])  # 设置mean和var
parse.add_argument('--SaltParams', type=float, nargs='+', default=[0.1, 1])  # 设置density 和概率
parse.add_argument('--device', help="训练装置")
parse.add_argument("--logger", help="日志对象")
args = parse.parse_args()


args.device = torch.device("cuda:0")
args.logger = logging.getLogger(name="n2n")
args.logger.setLevel(level=logging.DEBUG)


if __name__ == "__main__":


    trainer = Trainer(args)
    if args.varity:
        trainer.verify(1)
    else:
        trainer.run()
