import numpy as np
import torch
from scipy.spatial import distance

def create_affinity_graph_from_score(phenotypic_data, scores, threshold):
    '''

    :param phenotypic_data: .npy file    shape = [num of subjects, num of phenotypic elements]
    :return:
    '''
    dic = {'hand': 0, 'hurt': 1, 'pain': 2, 'sex': 3, 'age': 4, 'pressure': 5,
           'lipid': 6, 'diabetes': 7, 'smoke': 8, 'drink': 9, 'blood': 10}
    num_nodes = len(phenotypic_data)
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        ind = int(dic[l])
        if l == 'age':
            for k in range(num_nodes):
                for j in range(k+1, num_nodes):
                    try:
                        val = abs(float(phenotypic_data[k, ind]) - float(phenotypic_data[j, ind]))
                        if val < threshold:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing value
                        pass
        else:
            for k in range(num_nodes):
                for j in range(k+1, num_nodes):
                    if phenotypic_data[k, ind] == phenotypic_data[j, ind]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph

def get_static_affinity_adj(features, phenotypic, args):
    pd_affinity = create_affinity_graph_from_score(phenotypic, args.scores, args.threshold)
    distv = distance.pdist(features, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    adj = pd_affinity * feature_sim

    return adj
    # return pd_affinity 可直接返回性能有提升


def get_static_affinity_adj_test(features_test, phenotypic_test, args):
    pd_affinity_test = create_affinity_graph_from_score(phenotypic_test, args.scores, args.threshold)
    distv_test = distance.pdist(features_test, metric='correlation')
    dist_test = distance.squareform(distv_test)
    sigma_test = np.mean(dist_test)
    feature_sim_test = np.exp(- dist_test ** 2 / (2 * sigma_test ** 2))
    adj_test = pd_affinity_test * feature_sim_test

    return adj_test