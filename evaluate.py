from scipy import io
from sklearn.cluster import KMeans
from skfeature.utility import unsupervised_evaluation
from util import load_dataset
import nxmetis
# import pymetis
import numpy as np
import networkx as nx
import gc
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

NEW_DATASETS = ['Lung-Cancer', 'Movementlibras', 'Sonar']
NEW_DATASETS2 = ['waveform-5000']
NEW_DATASETS3 = ['UAV1', 'UAV2']
NEW_DATASETS6 = ['UJIndoorLoc']
DATASETS = ['COIL20', 'ORL', 'colon', 'madelon', 'Lung-Cancer', 'Movementlibras', 'Sonar', 'waveform-5000', 'UAV1', 'UAV2', 'UJIndoorLoc', 'nci9']

NUM_SELECT = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
SELECT_PER_CLUSTER = 1
OUTLIER_PERCENT = 0.1
SET_ZERO_PERCENT = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--alpha', type=float, required=False, default=1.0)
parser.add_argument('-b', '--beta', type=float, required=False, default=0.001)
args = parser.parse_args()



def main(DATASET, select_k):
    # load data
    X, y, NUM_CLASSES, num_features = load_dataset(DATASET)

    adj = np.loadtxt('results/' + DATASET + '_result_' + str(args.alpha) + '_' + str(args.beta) + '.csv', delimiter=",")
    # adj = np.cov(X.T)
    adj = np.abs(adj)
    selected_features = []
    select = np.zeros(num_features)
    count = 0
    num_selected = int(select_k * num_features)



    adj = adj - adj * np.eye(adj.shape[0])
    top_k = int(adj.shape[0] * adj.shape[0] - (adj.shape[0] * adj.shape[0] - adj.shape[0]) * (1 - SET_ZERO_PERCENT))
    median = np.sort(adj,axis=None)[top_k]
    # median = 0
    adj[adj <= median] = 0

    row_sum = np.sum(adj, axis=0)
    delete_list = []

    # method 1 for outlier
    for i in range(int(num_features * OUTLIER_PERCENT)):
        index = np.argmin(row_sum)
        row_sum[index] = np.inf
        delete_list.append(index)
        select[index] = -1

    # # method 2 for outlier
    # estimator = KMeans(n_clusters=10)
    # estimator.fit(row_sum[:, np.newaxis])
    # centroids = estimator.cluster_centers_
    # labels = estimator.labels_
    # index = np.argmin(centroids)
    # for i in range(num_features):
    #     if labels[i] == index:
    #         delete_list.append(i)

    delete_list.sort()
    adj = np.delete(adj, delete_list, axis=0)
    adj = np.delete(adj, delete_list, axis=1)


    adj_int = adj/np.max(adj)*1000
    adj_int = adj_int.astype(np.int)
    column = np.max(adj_int, axis=0)
    


    adj_int = nx.from_numpy_array(adj_int)
    (st, parts) = nxmetis.partition(adj_int, int((num_selected+SELECT_PER_CLUSTER-1) / SELECT_PER_CLUSTER))
    cluster_membership = {node: membership for node, membership in enumerate(parts)}

    # adjacency = np.arange(num_features)[np.newaxis,:].repeat(num_features,axis=0)
    # weight = adj_int.reshape(-1)
    # cutcount, part_vert = pymetis.part_graph(num_selected, adjacency=adjacency, eweights=weight)
    # cluster_membership = {}
    # for i, p in enumerate(set(part_vert)):
    #     ind = np.where(np.array(part_vert) == p)[0]
    #     cluster_membership[p] = ind

    # print("========================================================")
    for c in cluster_membership.keys():
        # print(c, cluster_membership[c])
        for _ in range(SELECT_PER_CLUSTER):
            pick, max_weight = 0, 0.0
            for i in range(len(cluster_membership[c])):
                if select[cluster_membership[c][i]] != 0:
                    continue
                cur_weight = 0
                for j in range(len(cluster_membership[c])):
                    cur_weight += adj[cluster_membership[c][i]][cluster_membership[c][j]]
                if max_weight < cur_weight:
                    max_weight = cur_weight
                    pick = i
                # if max_weight < column[i]:
                #     max_weight = column[i]
                #     pick = i

            if len(cluster_membership[c]) == 0:
                continue
            original_index = cluster_membership[c][pick]
            for i in delete_list:
                if original_index > i:
                    original_index += 1
                else:
                    break
            select[original_index] = 1
    # print("========================================================")





    # method 2 for feature selection
    # adj_int = adj_int - adj_int * np.eye(adj_int.shape[0])
    # while(count < num_selected):
    #     idx = np.where(adj_int == np.max(adj_int))
    #     original_index1 = idx[0][0]
    #     original_index2 = idx[1][0]
    #     if adj_int[idx[0][0]][idx[1][0]] == -1:
    #         break
    #     adj_int[idx[0][0]][idx[1][0]] = -1
    #     for i in delete_list:
    #         if original_index1 > i:
    #             original_index1 += 1
    #         else:
    #             break
    #     for i in delete_list:
    #         if original_index2 > i:
    #             original_index2 += 1
    #         else:
    #             break
    #     if select[original_index1] == 0:
    #         select[original_index1] = 1
    #         count += 1
    #         for j in range(len(adj_int)):
    #             adj_int[idx[0][0]][j] = -1
    #             adj_int[j][idx[0][0]] = -1
    #     if count >= num_selected:
    #         break
    #     if select[original_index2] == 0:
    #         select[original_index2] = 1
    #         count += 1
    #         for j in range(len(adj_int)):
    #             adj_int[idx[1][0]][j] = -1
    #             adj_int[j][idx[1][0]] = -1



    # method 3 for feature selection
    # while(count < num_selected):
    #     idx = np.where(column == np.max(column))
    #     original_index = idx[0][0]
    #     if column[original_index] == -1:
    #         break
    #     column[original_index] = -1
    #     for i in delete_list:
    #         if original_index > i:
    #             original_index += 1
    #         else:
    #             break
    #     if select[original_index] == 0:
    #         select[original_index] = 1
    #         count += 1



    X = np.array(X)
    X = X.T
    count = 0
    for i in range(num_features):
        if select[i] == 1:
            selected_features.append(X[i])
            count += 1
    selected_features = np.array(selected_features)
    selected_features = selected_features.T

    # perform kmeans clustering based on the selected features and repeats 20 times
    nmi_total = []
    acc_total = []
    for i in range(10):
        nmi, acc = unsupervised_evaluation.evaluation(X_selected=selected_features, n_clusters=NUM_CLASSES, y=y)
        nmi_total.append(nmi)
        acc_total.append(acc)


    # del X, y, y_set, adj, adj_int, parts, cluster_membership, select, selected_features
    # gc.collect()
    nmi_total = np.array(nmi_total)
    acc_total = np.array(acc_total)

    return np.mean(nmi_total), np.mean(acc_total), count, np.std(nmi_total), np.std(acc_total)


if __name__ == '__main__':
    for _ in range(1):
        res_acc, res_acc_std = [], []
        res_nmi, res_nmi_std = [], []
        start = time.time()
        now = time.time()
        for dataset in DATASETS:
            cur_acc, cur_nmi, cur_acc_std, cur_nmi_std = [], [], [], []
            for select_k in NUM_SELECT:
                nmi, acc, count, std_nmi, std = main(dataset, select_k)
                cur_acc.append(acc)
                cur_acc_std.append(std)
                cur_nmi.append(nmi)
                cur_nmi_std.append(std_nmi)
            res_acc.append(np.array(cur_acc))
            res_acc_std.append(np.array(cur_acc_std))
            res_nmi.append(np.array(cur_nmi))
            res_nmi_std.append(np.array(cur_nmi_std))
            print(dataset, '----Selected:', count, '----Time:', time.time() - now)
            now = time.time()
        res_acc = np.array(res_acc)
        res_acc_std = np.array(res_acc_std)
        res_nmi = np.array(res_nmi)
        res_nmi_std = np.array(res_nmi_std)
        np.savetxt('result_acc_' + str(args.alpha) + '_' + str(args.beta) + '.csv', res_acc, delimiter = ',')
        # np.savetxt('result_acc_std_' + str(args.alpha) + '_' + str(args.beta) + '.csv', res_acc_std, delimiter = ',')
        # np.savetxt('result_nmi_' + str(args.alpha) + '_' + str(args.beta) + '.csv', res_nmi, delimiter = ',')
        # np.savetxt('result_nmi_std_' + str(args.alpha) + '_' + str(args.beta) + '.csv', res_nmi_std, delimiter = ',')
        print("TOTAL TIME:", time.time() - start)