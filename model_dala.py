import torch
import torch.optim as optim
from models.layer import GraphConvolution
from torch.nn import Linear as Lin
import torch_geometric as tg
import torch.nn.utils as torch_utils
import torch.nn.functional as F
from torch import nn
from models.PAE import PAE
from opt_EVGCN import *
from dataloader_EVGCN import dataloader
from dala_loss import compute_loss_of_classes,DALA,get_num_class_list

opt = OptInit().initialize()
dl = dataloader(opt)
label_dir = './labels.npy'
phenotypic_dir = './phenotypic.npy'
feature_dir = './features.npy'
label_dir_test = './labels_test.npy'
phenotypic_dir_test = './phenotypic_test.npy'
feature_dir_test = './features_test.npy'
raw_features, y, nonimg,x = dl.load_data(label_dir, phenotypic_dir, feature_dir)

def my_sigmoid(mx, dim):
    mx = torch.sigmoid(mx)
    return F.normalize(mx, p=1, dim=dim)
def save_best_results(best_stats, class_prob, y_test):
    # 保存最优结果的输出和其对应的标签
    np.save('best_class_prob.npy', class_prob)
    np.save('best_y_test.npy', y_test)
class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg):
        super(Discriminator, self).__init__()
        K = 3
        hidden = [hgc for i in range(lg)]
        self.dropout = dropout
        self.edge_dropout = edge_dropout
        bias = False
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg
        self.gconv = nn.ModuleList()
        for i in range(lg):
            in_channels = input_dim if i == 0 else hidden[i-1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias))
        cls_input_dim = sum(hidden)

        self.cls = nn.Sequential(
            torch.nn.Linear(cls_input_dim, 256),
            torch.nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            torch.nn.Linear(256, num_classes)
        )

        self.edge_net = PAE(input_dim=edgenet_input_dim//2, dropout=dropout)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
        # self.edge_net.model_init()

    def forward(self, features, edge_index, edgenet_input, enforce_edropout=False):
        if self.edge_dropout > 0:
            if enforce_edropout or self.training:
                # edgenet_input = torch.tensor(edgenet_input)
                one_mask_1 = torch.ones([edgenet_input.shape[0], 1])
                one_mask = torch.tensor(one_mask_1)
                self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))
                edge_index = edge_index[:, self.bool_mask]
                edgenet_input = edgenet_input[self.bool_mask]



        edge_weight = torch.squeeze(self.edge_net(edgenet_input.cuda()))
        features = torch.tensor(features)
        features = F.dropout(features, self.dropout, self.training).to(opt.device)
        # edge_index = edge_index.float()
        features = features.float()
        h = self.relu(self.gconv[0](features, edge_index, edge_weight))
        h0 = h

        for i in range(1, self.lg):
            h = F.dropout(h, self.dropout, self.training)
            h = self.relu(self.gconv[i](h, edge_index, edge_weight))
            jk = torch.cat((h0, h), axis=1)
            h0 = jk
        logit = self.cls(jk)
        logit = F.log_softmax(logit, dim=1)
        return jk, logit
class Weighing(nn.Module):
    def __init__(self, nfeat, nhid, weighing_output_dim, dropout, order):
        super(Weighing, self).__init__()

        layers = []
        layers.append(GraphConvolution(nfeat, nhid[0], order=order))
        for i in range(len(nhid) - 1):
            layers.append(GraphConvolution(nhid[i], nhid[i + 1], order=order))

        layers.append(GraphConvolution(nhid[-1], weighing_output_dim, order=order))
        self.gc = nn.ModuleList(layers)

        self.dropout = dropout

    def forward(self, x, adj, func, samples=-1):
        end_layer = len(self.gc) - 1
        for i in range(end_layer):
            x = self.gc[i](x, adj)
            x = F.leaky_relu(x, inplace=False)
            x = F.dropout(x, self.dropout, training=self.training)

        if type(samples) is int:
            weights = func(self.gc[-1](x, adj), dim=0)
        else:
            weights = func(self.gc[-1](x, adj)[samples, :], dim=0)

        if len(weights.shape) == 1:
            weights = weights[None, :]

        return weights, x


class EVGCN(torch.nn.Module):
    def __init__(self, features,adj,n_ws,weighing_output_dim,gamma,struc_Ws,act,edge_index,edgenet_input):
        super(EVGCN, self).__init__()

        nfeat = features.shape[1]
        if type(adj['D']) is list:
            order_D = len(adj['D'])
        else:
            order_D = 1
        if type(adj['W'][0]) is list:
            order_W = len(adj['W'][0])
        else:
            order_W = 1
        self.n_ws = n_ws
        self.wod = weighing_output_dim
        self.edge_index = edge_index.cuda()
        self.edgenet_input = edgenet_input.cuda()
        self.net_D = Discriminator(nfeat, opt.num_classes, opt.dropout, edge_dropout=opt.edropout, hgc=opt.hgc, lg=opt.lg, edgenet_input_dim=2*nonimg.shape[1]).to(opt.device)
        self.gamma = gamma
        self.opt_D = optim.SGD(self.net_D.parameters(), lr=0.01, weight_decay=5e-4)
        self.net_Ws = []
        self.opt_Ws = []
        for i in range(n_ws):
            self.net_Ws.append(Weighing(nfeat=nfeat, nhid=struc_Ws[i]['nhid'], weighing_output_dim=weighing_output_dim
                              , dropout=struc_Ws[i]['dropout'], order=order_W))
            self.opt_Ws.append(
                optim.SGD(self.net_Ws[-1].parameters(), lr=struc_Ws[i]['lr'], weight_decay=struc_Ws[i]['wd']))
        self.adj_D = adj['D']
        self.adj_W = adj['W']
        self.features = features
        self.act = act

    def run_D(self, samples):
        class_prob_1, embed_1 = self.net_D(self.features, self.edge_index, self.edgenet_input)
        class_prob=class_prob_1[samples]
        embed = embed_1[samples]
        return class_prob, embed

    def run_W(self, samples, labels, args_cuda=False, equal_weights=False):
        batch_size = samples.shape[0]
        embed = None
        if equal_weights:
            max_label = int(labels.max().item() + 1)
            weight = torch.empty(batch_size)
            for i in range(max_label):
                labels_indices = (labels == i).nonzero().squeeze()
                if len(labels_indices.shape) == 0:
                    batch_size = 1
                else:
                    batch_size = len(labels_indices)
                if labels_indices is not None:
                    weight[labels_indices] = 1 / batch_size * torch.ones(batch_size)
            weight = weight / max_label
        else:
            max_label = int(labels.max().item() + 1)
            weight = torch.empty(batch_size)
            if args_cuda:
                weight = weight.cuda()
            for i in range(max_label):
                labels_indices = (labels == i).nonzero().squeeze()
                if labels_indices is not None:
                    sub_samples = samples[labels_indices]
                    weight_, embed = self.net_Ws[i](x=self.features[sub_samples,:], adj=self.adj_W[i], samples=-1, func=self.act)
                    weight[labels_indices] = weight_.squeeze() if self.wod == 1 else weight_[:,i]
            weight = weight / max_label

        if args_cuda:
            weight = weight.cuda()

        return weight, embed


    def loss_function_D(self, output, labels, weights):
        dala = -weights * (labels.float() * output).sum(1)
        weight_loss = torch.sum(-weights * (labels.float() * output).sum(1), -1)
        return weight_loss,dala

    def loss_function_G(self, output, labels, weights):
        return torch.sum(- weights * (labels.float() * output).sum(1), -1) - self.gamma*torch.sum(weights*torch.log(weights+1e-20))

    def zero_grad_both(self):
        self.opt_D.zero_grad()
        for opt in self.opt_Ws:
            opt.zero_grad()


    def run_both(self,epoch_for_D, epoch_for_W, labels_one_hot,idx_train,labels, adj_1,model,features,edge_index,edgenet_input, samples=-1,
                 args_cuda=False, equal_weights=False):
        labels_not_onehot = labels_one_hot.max(1)[1].type_as(labels_one_hot)
        for e_D in range(epoch_for_D):
            self.zero_grad_both()

            # class_prob_1, embed = self.run_D(samples)
            embed, class_prob_1 = self.run_D(samples)

            weights_1, _ = self.run_W(samples=samples, labels=labels_not_onehot, args_cuda=args_cuda, equal_weights=equal_weights)

            loss_D,dala_loss = self.loss_function_D(output=class_prob_1, labels=labels_one_hot, weights=weights_1)
            cls_num_list = get_num_class_list(idxs=idx_train, labels=y)
            cls_loss = compute_loss_of_classes(adj=adj_1['D'][idx_train][:, idx_train],
                                               model=model.net_D, model_1=model, x=features,
                                               label=labels[idx_train], id=idx_train,
                                               n_classes=2, class_prob=class_prob_1, labels_one_hot=labels_one_hot,
                                               weights=weights_1, edge_index=edge_index, edgenet_input=edgenet_input)
            ce_criterion = DALA(cls_num_list=cls_num_list, cls_loss=cls_loss)
            loss = ce_criterion(x=class_prob_1, target=labels_one_hot,weights=weights_1,gamma=self.gamma)
            # loss_function_D(cls_num_list=cls_num_list, cls_loss=cls_loss, output=class_prob,
            #                 labels=labels_one_hot[idx_train], weights=weights, epoch=epoch).item()
            loss.backward()
            # max_grad_norm = 1.0  # 设置裁剪的最大范数
            # torch_utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            self.opt_D.step()

        for e_W in range(epoch_for_W):
            self.zero_grad_both()

            # class_prob_2, embed = self.run_D(samples)
            embed,class_prob_2 = self.run_D(samples)
            weights_2, embeds = self.run_W(samples=samples, labels=labels_not_onehot, args_cuda=args_cuda, equal_weights=equal_weights)

            loss_G = -self.loss_function_G(output=class_prob_2, labels=labels_one_hot, weights=weights_2)
            loss_G.backward()
            for opt in self.opt_Ws:
                opt.step()

    def train(self):
        self.net_D.train()
        for i in range(self.n_ws):
            self.net_Ws[i].train()

    def eval(self):
        self.net_D.eval()
        for i in range(self.n_ws):
            self.net_Ws[i].eval()

    def cuda(self):
        self.features = self.features.to(device='cuda')
        for i in range(len(self.adj_D)):
            self.adj_D[i] = self.adj_D[i].to(device='cuda')
        for i in range(len(self.adj_W)):
            self.adj_W[i] = self.adj_W[i].to(device='cuda')
        self.net_D.cuda()
        for i in range(self.n_ws):
            self.net_Ws[i].cuda()