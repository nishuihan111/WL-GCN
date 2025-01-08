import getMatrix
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
class MyLoss(nn.Module):
    def __init__(self, classes):
        super(MyLoss, self).__init__()
        self.classes = classes

        self.dim = 64
        self.centers = torch.zeros((classes, self.dim)).cuda()
        self.maxR = torch.zeros((classes,)).cuda()
        self.maxf = torch.zeros((classes, classes)).cuda()
        self.cnt = 0
    def generatelabels(self, size, real_labels=None):
        x = torch.Tensor(torch.zeros((size, self.classes), requires_grad=True)).cuda()
        if real_labels is None:  # 生成随机标签
            y = [np.random.randint(0, 9) for i in range(size)]
            x[np.arange(size), y] = 1
        else:
            x[np.arange(size), real_labels] = 1

        return x
    def forward(self, y_pred, target, weight, out, epoch, epochs, nums_cls_list):
        y_true = self.generatelabels(target.shape[0], target.long())
        num_batch = torch.sum(y_true, 0)
        x_means = []
        maxs = torch.zeros((self.classes,)).cuda()
        rs = torch.zeros((self.classes,)).cuda()

        inter_dis = 0
        inter_cnt = 0
        intra_cnt = 0
        intra_dis = 0
        if torch.any(torch.isnan(y_pred)):
            print('error')
            exit()
        for i in range(self.classes):
            if num_batch[i] == 0:
                continue
            ind = torch.where(target == i)[0]
            x = out[ind]
            xmean = torch.mean(x.detach(), 0)

            xmean = xmean.cuda()
            self.centers[i] = self.centers[i] * 0.1 + 0.9 * xmean

            x = x.cuda()
            xsum = torch.sqrt(torch.sum((x.detach() - self.centers[i]) ** 2, 1))
            rs[i] = max(xsum)

            xmax = max(xsum)
            maxs[i] = xmax

        i = 0
        while i < self.classes:
            if maxs[i] > 0:
                break
            i += 1
        maxx = maxs[i]
        mcs = torch.ones((self.classes,)).cuda()

        ncl = torch.Tensor(nums_cls_list).cuda()
        ncl = 10 * ncl / ncl[0]

        maxs = maxs / maxx
        maxs[maxs > 1] = 1
        if self.classes > 10:
            if torch.all(self.maxR == 0):
                self.maxR = maxs
            else:
                for i in range(self.classes):
                    if maxs[i] != 0:
                        self.maxR[i] = self.maxR[i] * 0.1 + maxs[i] * 0.9
            maxs = self.maxR

        points = out
        p1 = points.repeat(points.shape[0], 1)
        p2 = points.repeat(1, points.shape[0]).reshape(p1.shape[0], -1)
        dis_mat = torch.sum((p1 - p2) ** 2, 1).reshape(points.shape[0], -1)
        dis_m = torch.zeros((dis_mat.shape[0], dis_mat.shape[0], dis_mat.shape[0])).cuda()
        target_mat = torch.zeros((dis_mat.shape[0], dis_mat.shape[0])).cuda()

        a = dis_mat.repeat(dis_mat.shape[0], 1)
        b = dis_mat.repeat(1, dis_mat.shape[0]).reshape(a.shape[0], -1)
        c = a + b
        c = c.reshape((dis_m.shape))
        dis_m = c

        target_d = target.unsqueeze(0)
        t1 = target_d.repeat(target_d.shape[1], 1)
        target_mat = t1 - t1.t()
        gamma = 2
        # 超参需要调整
        if self.classes == 100:
            gamma = 8
        elif self.classes == 200:
            gamma = 15
        elif self.classes == 365:
            gamma = 5
        alpha = (epoch / epochs) ** gamma
        # alpha = 1-0.5*np.cos(epoch/epochs*np.pi) - 0.5
        if self.classes == 2:
            sigma = -30
            lambd = 0.5
        elif self.classes == 100:
            sigma = -50
            lambd = 0.5
        elif self.classes == 200:
            sigma = -150
            lambd = 0.3
        elif self.classes == 365:
            sigma = -10
            lambd = 0.5
        dis_m = dis_m.permute(0, 2, 1)
        judge_mat = (dis_m >= dis_mat) + 0
        judge_mask = torch.sum(judge_mat, 1)
        judge_mask = (judge_mask == dis_m.shape[0]) + 0
        judge_mask = judge_mask.float().cuda()
        d = torch.abs((torch.eye(dis_m.shape[0]) - 1).cuda())

        judge_mask = torch.mul(d, judge_mask)
        target_mask = (target_mat != 0) + 0
        target_mask = target_mask.float().cuda()
        mask = torch.mul(target_mask, judge_mask)
        inter_dis = torch.exp(dis_mat / sigma)
        cnt = mask.sum()

        intra_dis = torch.sum((points - weight[target.long()]) ** 2, 1)
        # intra_dis = torch.sum((points - self.centers[target]) ** 2, 1)
        intra_dis = torch.exp(intra_dis / sigma)
        intra_dis = intra_dis.unsqueeze(0)
        intra_dis = intra_dis.repeat(inter_dis.shape[0], 1)
        lossdis = inter_dis - intra_dis + lambd
        lossdis = lossdis.cuda()
        inter_dis_new = torch.mul(mask, lossdis)
        lossdis[lossdis < 0] = 0
        inter_dis_new[inter_dis_new < 0] = 0
        mcs = ncl
        scale = mcs[target.long()]
        scale = scale.unsqueeze(0)
        scale = scale.repeat((inter_dis.shape[0], 1))
        inter_dis_new /= scale
        alpha = torch.tensor(alpha)
        alpha = alpha.cuda()
        y_pred =y_pred.cuda()
        y_pred *= maxs ** alpha
        x = torch.softmax(y_pred, 1)
        y = torch.log(x ** y_true)
        if torch.any(torch.isnan(y)):
            print(inter_dis)
            print(torch.any(torch.isnan(y_pred)))
            exit()
        beta = max((1 - alpha), 0.7)
        loss = beta * inter_dis_new.sum() / cnt
        return loss

    def updateC(self):
        self.centers = torch.zeros((self.classes, self.dim)).cuda()
        self.maxR = torch.zeros((self.classes,)).cuda()
        self.maxf = torch.zeros((self.classes, self.classes)).cuda()
class dataloader():
    def __init__(self, opt):
        self.args = opt
        self.num_classes = opt.num_classes
        self.cls_num = 2

    def load_data(self, label_dir, phenotypic_dir, feature_dir):
        dic = {'hand': 0, 'hurt': 1, 'pain': 2, 'sex': 3, 'age': 4, 'pressure': 5,
               'lipid': 6, 'diabetes': 7, 'smoke': 8, 'drink': 9, 'blood': 10}

        labels = np.load(label_dir, allow_pickle=True)
        phenotypic = np.load(phenotypic_dir, allow_pickle=True)
        # num of subjects
        num_nodes = len(labels)
        # num of phenotypic elements
        num = len(self.args.scores)
        #生成num_nodes行num列的0矩阵，类型为float32
        phonetic_data = np.zeros([num_nodes, num], dtype=np.float32)
        for k, score in enumerate(self.args.scores):
            ind = dic[score]
            ph = phenotypic[:, ind]
            tmp = np.zeros([num_nodes], dtype=np.float32)
            if score in ['hand', 'sex', 'blood']:
                unique = np.unique(list(ph)).tolist()
                for i in range(num_nodes):
                    tmp[i] = unique.index(ph[i])
            else:
                for i in range(num_nodes):
                    tmp[i] = float(ph[i])

            phonetic_data[:, k] = tmp

        self.y = labels
        self.y = self.y.astype(np.int64)
        self.raw_features = np.load(feature_dir, allow_pickle=True)
        self.raw_features = self.raw_features.astype(float)
        self.phenotypic = phonetic_data
        self.x = np.hstack((self.phenotypic[:,:],self.raw_features))
        return self.raw_features, self.y, self.phenotypic, self.x

    def load_data_test(self, label_dir_test, phenotypic_dir_test, feature_dir_test):
            dic = {'hand': 0, 'hurt': 1, 'pain': 2, 'sex': 3, 'age': 4, 'pressure': 5,
                   'lipid': 6, 'diabetes': 7, 'smoke': 8, 'drink': 9, 'blood': 10}

            labels_test = np.load(label_dir_test, allow_pickle=True)
            phenotypic_test = np.load(phenotypic_dir_test, allow_pickle=True)
            # num of subjects
            num_nodes_test = len(labels_test)
            # num of phenotypic elements
            num_test = len(self.args.scores)
            # 生成num_nodes行num列的0矩阵，类型为float32
            phonetic_data_test = np.zeros([num_nodes_test, num_test], dtype=np.float32)
            for k, score in enumerate(self.args.scores):
                ind_test = dic[score]
                ph_test = phenotypic_test[:, ind_test]
                tmp_test = np.zeros([num_nodes_test], dtype=np.float32)
                if score in ['hand', 'sex', 'blood']:
                    unique_test = np.unique(list(ph_test)).tolist()
                    for i in range(num_nodes_test):
                        tmp_test[i] = unique_test.index(ph_test[i])
                else:
                    for i in range(num_nodes_test):
                        tmp_test[i] = float(ph_test[i])

                phonetic_data_test[:, k] = tmp_test

            self.y_test = labels_test
            self.y_test = self.y_test.astype(np.int64)
            self.raw_features_test = np.load(feature_dir_test, allow_pickle=True)
            self.raw_features_test = self.raw_features_test.astype(float)
            self.phenotypic_test = phonetic_data_test
            self.x_test = np.hstack((self.phenotypic_test[:, :], self.raw_features_test))
            return self.raw_features_test, self.y_test, self.phenotypic_test, self.x_test

    def data_split(self, n_folds):

        skf1 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=111)
        skf2 = StratifiedKFold(n_splits=n_folds, shuffle=False)
        train_index, val_index, test_index = [], [], []
        # self.raw_features = self.raw_features[:414,:]
        # self.y = self.y[:414]
        for tr_ind, te_ind in skf1.split(self.raw_features, self.y):
            test_index.append(te_ind)
            for tr_tmp, val_tmp in skf2.split(self.raw_features[tr_ind], self.y[tr_ind]):
                train_index.append(tr_ind[tr_tmp])
                val_index.append(tr_ind[val_tmp])
                break
        return train_index, val_index, test_index


    def get_adj(self, scores, nonimg):
        n = self.x.shape[0]
        # num_edge = n*(1+n)//2 - n
        num_edge = n * (n-1) // 2
        pd_ftr_dim = nonimg.shape[1]
        edgenet_input = np.zeros([num_edge, 2 * pd_ftr_dim], dtype=np.float32)
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        aff_adj = getMatrix.get_static_affinity_adj(self.raw_features, self.phenotypic, self.args)
        flatten_ind = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_index[:, flatten_ind] = [i,j]
                edgenet_input[flatten_ind] = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i, j]
                flatten_ind += 1

        assert flatten_ind == num_edge

        keep_ind = np.where(aff_score > 0)[0]
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input,aff_adj
        # return edge_index, edgenet_input

class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)