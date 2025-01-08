import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class DALA(nn.Module):
    def __init__(self, cls_num_list, cls_loss, tau=1, weight=None, args=None):
        super(DALA, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        cls_loss = cls_loss.cuda()
        # cls_loss = torch.abs(cls_loss)
        # min_loss = torch.min(cls_loss)
        # if min_loss < 0:
        #     cls_loss = cls_loss - min_loss
        t = cls_p_list / (torch.pow(cls_loss, 0.25) + 1e-5)
        m_list = tau * torch.log(t)

        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target,weights,gamma):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight) - gamma*torch.sum(weights*torch.log(weights+1e-20))

def get_num_class_list(idxs, labels):
    n_classes = 2
    class_num = np.array([0] * n_classes)
    for idx in idxs:
        label = labels[idx]
        class_num[label] += 1
    return class_num.tolist()

def compute_loss_of_classes(adj, model, model_1, x, label, n_classes, class_prob, labels_one_hot, weights, id,
                            edge_index, edgenet_input):
    # criterion = loss_function_D()
    model.eval()

    loss_class = torch.zeros(n_classes).float()
    loss_list = []
    label_list = []

    with torch.no_grad():
        x, label = x.cuda(), label.cuda()
        _, logits_1 = model(x, edge_index=edge_index, edgenet_input=edgenet_input)
        logits = logits_1[id]
        _, dala_loss = model_1.loss_function_D(output=logits, labels=labels_one_hot, weights=weights)
        loss_list.append(dala_loss)
        label_list.append(label)

    loss_list = torch.cat(loss_list).cpu()
    label_list = torch.cat(label_list).cpu()

    for i in range(n_classes):
        idx = torch.where(label_list == i)[0]
        loss_class[i] = loss_list[idx].sum()

    return loss_class
import matplotlib.pyplot as plt

def visualize_features(features, labels, title):
    class_0_features = features[labels == 0]
    class_1_features = features[labels == 1]

    plt.figure()
    plt.scatter(class_0_features[:, 0], class_0_features[:, 1], color='blue', label='Class 0')
    plt.scatter(class_1_features[:, 0], class_1_features[:, 1], color='red', label='Class 1')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()