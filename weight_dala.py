import scipy.sparse as sp
from opt_EVGCN import *
from dataloader_EVGCN import dataloader
import networkx as nx
from util import normalize,fill_features
import torch
from sklearn.metrics import accuracy_score
from dala_loss import DALA
from dala_loss import compute_loss_of_classes
import numpy as np
import torch.nn.functional as F
from dala_loss import get_num_class_list
from model_dala import EVGCN
from util import accuracy, encode_onehot_torch, class_f1, auc_score
from model_dala import my_sigmoid

class_prob_results = []


def train(fold):
    struc_Ws = n_ws*[{'dropout': args.dropout_W,  'wd': 5e-4, 'lr': args.lr_W, 'nhid': structure_W}]
    stats = dict()
    act = my_sigmoid
    model = EVGCN(adj=adj_1, features=features, weighing_output_dim=1, act=act, gamma=args.gamma, struc_Ws=struc_Ws, n_ws=n_ws, edge_index=edge_index.cpu(), edgenet_input =edgenet_input.cpu())

    if use_cuda:
        model.cuda()
    best_class_prob_test = None  # 保存最好的测试集结果

    max_val = dict()
    max_val['f1Macro_val'] = 0
    max_test = dict()
    max_test['f1Macro_test'] = 0
    for epoch in range(args.epochs):
        model.train()
        model.run_both(model=model,features=features,labels=labels,edge_index=edge_index,edgenet_input=edgenet_input, adj_1=adj_1,idx_train=idx_train,epoch_for_D=args.epoch_D, epoch_for_W=args.epoch_W, labels_one_hot=labels_one_hot[idx_train, :],samples=idx_train, args_cuda=use_cuda, equal_weights=False)

        model.eval()
        class_prob, embed = model.run_D(samples=idx_train)
        weights, _ = model.run_W(samples=idx_train, labels=labels[idx_train], args_cuda=use_cuda, equal_weights=False)
        cls_num_list = get_num_class_list(idxs=idx_train, labels=y)
        cls_loss = compute_loss_of_classes(adj=adj_1['D'][idx_train][:, idx_train],
                                           model=model.net_D, model_1=model, x=features,
                                           label=labels[idx_train],id=idx_train,
                                           n_classes=2, class_prob=embed, labels_one_hot=labels_one_hot[idx_train],
                                           weights=weights,edge_index=edge_index,edgenet_input=edgenet_input)
        ce_criterion = DALA(cls_num_list=cls_num_list, cls_loss=cls_loss)
        stats['loss_train'] = ce_criterion(x=embed, target=labels_one_hot[idx_train],weights=weights,gamma=args.gamma).item()
        stats['nll_train'] = F.nll_loss(embed, labels[idx_train]).item()
        stats['acc_train'] = accuracy_score(np.argmax(labels_one_hot[idx_train].detach().cpu().numpy(), axis=1),np.argmax(embed.detach().cpu().numpy(), axis=1)).item()
        stats['f1Macro_train'] = class_f1(embed, labels[idx_train], type='macro')
        if nclass == 2:
            stats['f1Binary_train'] = class_f1(embed, labels[idx_train], type='binary', pos_label=pos_label)
            stats['AUC_train'] = auc_score(embed, labels_one_hot[idx_train])

        # best_class_prob_test = test(model, stats,epoch,best_class_prob_test,max_test)
        if epoch > drop_epochs and max_val['f1Macro_val'] < stats['f1Macro_val']:
            for key, val in stats.items():
                max_val[key] = val
        if epoch > drop_epochs and max_test['f1Macro_test'] < stats['f1Macro_test']:
            for key, val in stats.items():
                max_test[key] = val


        print('Epoch: {:04d}'.format(epoch + 1))
        print('acc_train: {:.4f}'.format(stats['acc_train']))
        print('f1_macro_train: {:.4f}'.format(stats['f1Macro_train']))
        print('loss_train: {:.4f}'.format(stats['loss_train']))


    # if best_class_prob_test is not None:
    #     np.save(f'best_class_prob_test_{fold+1}.npy', best_class_prob_test.detach().cpu().numpy())
    print('========Results==========')
    for key, val in max_val.items():
        if 'loss' in key or 'nll' in key or 'test' not in key:
            continue
        print(key.replace('_', ' ') + ' : ' + str(val))


def test(model, stats,epoch,best_class_prob_test,max_test):
    model.eval()
    class_prob,embed = model.run_D(samples=idx_val)
    weights, _ = model.run_W(samples=idx_val, labels=labels[idx_val], args_cuda=use_cuda, equal_weights=True)
    cls_num_list_val = get_num_class_list(idxs=idx_val, labels=y)
    cls_loss_val = compute_loss_of_classes(adj=adj_1['D'][idx_val][:, idx_val],
                                       model=model.net_D, model_1=model, x=features_cuda,
                                       label=labels[idx_val],id=idx_val,
                                       n_classes=2, class_prob=class_prob, labels_one_hot=labels_one_hot[idx_val],
                                       weights=weights,edge_index=edge_index,edgenet_input=edgenet_input)
    ce_criterion_val = DALA(cls_num_list=cls_num_list_val, cls_loss=cls_loss_val)
    stats['loss_val'] = ce_criterion_val(x=embed, target=labels_one_hot[idx_val],weights=weights,gamma=args.gamma).item()
    stats['nll_val'] = F.nll_loss(embed, labels[idx_val]).item()
    stats['acc_val'] = accuracy_score(np.argmax(labels_one_hot[idx_val].detach().cpu().numpy(), axis=1),np.argmax(embed.detach().cpu().numpy(), axis=1)).item()
    stats['f1Macro_val'] = class_f1(embed, labels[idx_val], type='macro')
    if nclass == 2:
        stats['f1Binary_val'] = class_f1(embed, labels[idx_val], type='binary', pos_label=pos_label)
        stats['AUC_val'] = auc_score(embed, labels_one_hot[idx_val])
    embed, class_prob_test = model.run_D(samples=idx_test)


    weights, _ = model.run_W(samples=idx_test, labels=labels[idx_test], args_cuda=use_cuda, equal_weights=True)
    cls_num_list_test = get_num_class_list(idxs=idx_test, labels=y)
    cls_loss_test = compute_loss_of_classes(adj=adj_1['D'][idx_test][:, idx_test],
                                            model=model.net_D, model_1=model, x=features_cuda,
                                            label=labels[idx_test],id=idx_test,
                                            n_classes=2, class_prob=class_prob_test, labels_one_hot=labels_one_hot[idx_test],
                                            weights=weights,edge_index=edge_index,edgenet_input=edgenet_input)
    ce_criterion_test = DALA(cls_num_list=cls_num_list_test, cls_loss=cls_loss_test)
    stats['loss_test'] = ce_criterion_test(x=class_prob_test, target=labels_one_hot[idx_test],weights=weights,gamma=args.gamma).item()
    stats['nll_test'] = F.nll_loss(class_prob_test, labels[idx_test]).item()
    stats['acc_test'] = accuracy_score(np.argmax(labels_one_hot[idx_test].detach().cpu().numpy(), axis=1),np.argmax(embed.detach().cpu().numpy(), axis=1)).item()
    stats['f1Macro_test'] = class_f1(class_prob_test, labels[idx_test], type='macro')

    if nclass == 2:
        stats['f1Binary_test'] = class_f1(class_prob_test, labels[idx_test], type='binary', pos_label=pos_label)
        stats['AUC_test'] = auc_score(class_prob_test, labels_one_hot[idx_test])
    # if best_class_prob_test is None or stats['f1Macro_test'] > max_test['f1Macro_test']:
    #     best_class_prob_test = class_prob_test.clone()
    #     for i in range (n_folds):
    #         best_labels_test = y[test_index[i]].copy()
    #         # np.save(f'best_labels_test_{i+1}.npy', best_labels_test)
    #
    # return best_class_prob_test
    # 在test函数中调用save_best_results()时传递class_prob和对应的标签y_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of epochs to train.')
    parser.add_argument('--epoch_D', type=int, default=1,
                        help='Number of training loop for discriminator in each epoch.')
    parser.add_argument('--epoch_W', type=int, default=1,
                        help='Number of training loop for discriminator in each epoch.')
    parser.add_argument('--lr_D', type=float, default=0.01,
                        help='Learning rate for discriminator.')
    parser.add_argument('--lr_W', type=float, default=0.01,
                        help='Equal learning rate for weighting networks.')
    parser.add_argument('--dropout_D', type=float, default=0.5,
                        help='Dropout rate for discriminator.')
    parser.add_argument('--dropout_W', type=float, default=0.5,
                        help='Dropout rate for weighting networks.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Coefficient of entropy term in loss function.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    ### This list shows the number of hidden neurons in each hidden layer of discriminator
    structure_D = [4]
    ### This list shows the number of hidden neurons in each hidden layer of weighting networks
    structure_W = [4]
    ### The results of first drop_epochs will be dropped for choosing the best network based on the validation set
    drop_epochs = 200
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    opt = OptInit().initialize()
    print('Loading dataset ...')
    dl = dataloader(opt)
    label_dir = './labels.npy'
    phenotypic_dir = './phenotypic.npy'
    feature_dir = './features.npy'
    label_dir_test = './labels_test.npy'
    phenotypic_dir_test = './phenotypic_test.npy'
    feature_dir_test = './features_test.npy'
    raw_features, y, nonimg,x = dl.load_data(label_dir, phenotypic_dir, feature_dir)
    raw_features_test, y_test, nonimg_test,x_test = dl.load_data_test(label_dir_test, phenotypic_dir_test, feature_dir_test)


    scores = opt.scores
    n_folds = 5
    train_index, val_index= dl.data_split(n_folds)
    test_index = np.arange(0,148)

    corrects = np.zeros(n_folds, dtype=np.int32)
    train_num = 0
    val_num = 0
    test_num = 0
    for fold in range(n_folds):
        print("\r\n============================== Fold {} =================================".format(fold))
        idx_train = train_index[fold]
        idx_val = val_index[fold]
        idx_test = test_index
        # idx_test = test_index[0]
        val_num += len(idx_val)
        test_num += len(idx_test)
        # print('Constructing graph data...')
        node_ftr = torch.tensor(raw_features)
        node_ftr_test = torch.tensor(raw_features_test)


        edge_index, edgenet_input, adj_1 = dl.get_adj(scores, nonimg)
        # adj_2 = adj_3[:,idx_train]
        # adj_1 = adj_2[idx_train, :]
        # edge_index_test, edgenet_input_test,adj_test = dl.get_adj_test(scores, nonimg_test)

        edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
        # edgenet_input_test = (edgenet_input_test - edgenet_input_test.mean(axis=0)) / edgenet_input_test.std(axis=0)

        # model = EV_GCN(node_ftr.shape[1], opt.num_classes, opt.dropout, edge_dropout=opt.edropout, hgc=opt.hgc, lg=opt.lg, edgenet_input_dim=2*nonimg.shape[1]).to(opt.device)
        # model = model.to(opt.device)

        # loss_fn = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(opt.device).detach()
        # features_cuda_test = torch.tensor(node_ftr_test, dtype=torch.float32).to(opt.device).detach()

        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        # edge_index_test = torch.tensor(edge_index_test, dtype=torch.long).to(opt.device)

        edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)
        # edgenet_input_test = torch.tensor(edgenet_input_test, dtype=torch.float32).to(opt.device)

        labels = torch.tensor(y, dtype=torch.long).to(opt.device)
        # labels_test = torch.tensor(y_test, dtype=torch.long).to(opt.device)

        n_node = adj_1.shape[1]#多少个节点
        adj_1 = nx.adjacency_matrix(nx.from_numpy_array(adj_1))
        # adj_test = nx.adjacency_matrix(nx.from_numpy_array(adj_test))

        # nx.from_numpy_array(adj) 将该邻接矩阵转换为 NetworkX 图对象，然后 nx.adjacency_matrix() 函数将该图对象转换为一个稀疏的邻接矩阵。
        adj_1 = adj_1 + adj_1.T.multiply(adj_1.T > adj_1) - adj_1.multiply(adj_1.T > adj_1)
        # adj_test = adj_test + adj_test.T.multiply(adj_test.T > adj_test) - adj_test.multiply(adj_test.T > adj_test)

        adj_D = normalize(adj_1 + sp.eye(adj_1.shape[0]))
        # adj_D_test = normalize(adj_test + sp.eye(adj_test.shape[0]))

        adj_W = normalize(adj_1 + sp.eye(adj_1.shape[0]))
        # adj_W_test = normalize(adj_test + sp.eye(adj_test.shape[0]))

        adj_1= dict()
        # adj_test= dict()

        adj_1['D'] = adj_D
        # adj_test['D'] = adj_D_test

        features_np = features_cuda.cpu().numpy()
        features = features_np.astype(np.float64)
        features = fill_features(features)

        # features_np_test = features_cuda_test.cpu().numpy()
        # features_test = features_np_test.astype(np.float)
        # features_test = fill_features(features_test)


        adj_1['W'] = []
        # adj_test['W'] = []

        labels = labels.cpu()
        # labels_test = labels_test.cpu()
        # labels_test = labels_test.cpu()
        for nc in range(labels.max() + 1):
            nc_idx = np.where(labels[idx_train] == nc)[0]
            nc_idx = np.array(idx_train)[nc_idx]
            adj_1['W'].append(adj_W[np.ix_(nc_idx, nc_idx)])

        features = torch.FloatTensor(features)
        for key,val in adj_1.items():
            if key == 'D':
                adj_1[key] = torch.FloatTensor(np.array(adj_1[key].todense()))#sparse_mx_to_torch_sparse_tensor(adj[i])
                # adj[key] = torch.FloatTensor(np.array(adj[key]))#sparse_mx_to_torch_sparse_tensor(adj[i])
            else:
                for i in range(len(val)):
                    adj_1[key][i] = torch.FloatTensor(np.array(adj_1[key][i].todense()))

        labels = torch.LongTensor(labels)
        # labels_test  = torch.LongTensor(labels_test)
        idx_train = torch.LongTensor(idx_train).cuda()
        idx_val = torch.LongTensor(idx_val).cuda()
        idx_test = torch.LongTensor(idx_test).cuda()
        ### start of code
        labels_one_hot = encode_onehot_torch(labels)
        nclass = labels_one_hot.shape[1]
        n_ws = nclass
        pos_label = None
        if nclass == 2:
            pos_label = 1
            zero_class = (labels == 0).sum()
            one_class = (labels == 1).sum()
            if zero_class < one_class:
                pos_label = 0
        if use_cuda:
            for key, val in adj_1.items():
                if type(val) is list:
                    for i in range(len(adj_1)):
                        adj_1[key][i] = adj_1[key][i].cuda()
                else:
                    adj_1[key] = adj_1[key].cuda()
            features = features.cuda()
            labels_one_hot = labels_one_hot.cuda()
            labels = labels.cuda()
            # labels_one_hot_test = labels_one_hot_test.cuda()
            # labels_test = labels_test.cuda()

        ### Training the networks
        train(fold)


