import datetime
import argparse
import random
import numpy as np
import torch

class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='HMGCN')
        parser.add_argument('--train', default=0, type=int)
        parser.add_argument('--use_cpu', action='store_false')
        parser.add_argument('--hgc', type=int, default=16)
        parser.add_argument('--lg', type=int, default=4)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--wd', default=5e-5, type=float)
        parser.add_argument('--edropout', default=0.3)
        parser.add_argument('--dropout', default=0.5, type=float)
        parser.add_argument('--scores', default=['hand', 'hurt', 'pain', 'sex', 'age', 'pressure',
                                'lipid', 'diabetes', 'smoke', 'drink', 'blood'])
        parser.add_argument('--threshold', type=int, default=2)#阈值分为几类
        parser.add_argument('--ckpt_path', type=str, default='./save_models')
        parser.add_argument('--warmup', type=int, default=1,help='warmup')
        parser.add_argument('--imb_ratio', type=float, default=10,help='Imbalance Ratio')
        parser.add_argument('--loss_type', type=str, default='ce',
                            help='Loss type')
        parser.add_argument('--keep_prob', type=float, default=0.01,
                            help='Keeping Probability')
        parser.add_argument('--pred_temp', type=float, default=2,
                            help='Prediction temperature')
        parser.add_argument('--net', type=str, default='EV_GCN',
                            help='Architecture name')
        parser.add_argument('--n_layer', type=int, default=2,
                            help='the number of layers')
        parser.add_argument('--feat_dim', type=int, default=256,
                            help='Feature dimension')
        # GAT
        parser.add_argument('--n_head', type=int, default=2,
                            help='the number of heads in GAT')
        args = parser.parse_args()
        args.time = datetime.datetime.now().strftime("%y%m%d")

        # if args.use_cpu:
        #     args.device = torch.device('cpu')
        # else:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args

    def print_args(self):
        print("==========          CONFIG          ==========")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========        CONFIG END        ==========")
        print("\n")
        phase = 'train' if self.args.train == 1 else 'eval'
        print("===> Phase is {}".format(phase))

    def initialize(self):
        self.set_seed(123)
        self.print_args()
        return self.args

    def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


