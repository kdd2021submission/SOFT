from scipy import io
from sklearn.cluster import KMeans
from skfeature.utility import unsupervised_evaluation
from util import load_dataset
import nxmetis
import pymetis
import numpy as np
import networkx as nx
import gc
import time
import warnings
warnings.filterwarnings("ignore")

NEW_DATASETS = ['Lung-Cancer', 'Movementlibras', 'Sonar']
NEW_DATASETS2 = ['waveform-5000']
NEW_DATASETS3 = ['UAV1', 'UAV2']
NEW_DATASETS6 = ['UJIndoorLoc']
DATASETS = ['COIL20', 'ORL', 'colon', 'madelon', 'Lung-Cancer', 'Movementlibras', 'Sonar', 'waveform-5000', 'UAV1', 'UAV2', 'UJIndoorLoc', 'nci9']

NUM_SELECT = [0.1]#[0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
SELECT_PER_CLUSTER = 1



def main(DATASET, select_k):
    # load data
    X, y, NUM_CLASSES, num_features = load_dataset(DATASET)

    adj = np.loadtxt('results/' + DATASET + '_result.csv', delimiter=",")

    selected_features = []
    select = np.zeros(num_features)
    count = 0
    num_selected = int(select_k * num_features)
  
    while(count < num_selected):
        idx = np.where(adj == np.max(adj))
        original_index = idx[0][0]
        if adj[original_index] == -999:
            break

        adj[original_index] = -999
        if select[original_index] == 0:
            select[original_index] = 1
            count += 1



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
    for index in range(1):
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
        np.savetxt('result_acc_' + str(index) + '.csv', res_acc, delimiter = ',')
        np.savetxt('result_acc_std_' + str(index) + '.csv', res_acc_std, delimiter = ',')
        np.savetxt('result_nmi_' + str(index) + '.csv', res_nmi, delimiter = ',')
        np.savetxt('result_nmi_std_' + str(index) + '.csv', res_nmi_std, delimiter = ',')
        print("TOTAL TIME:", time.time() - start)