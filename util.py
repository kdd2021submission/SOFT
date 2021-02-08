import os
import pickle
import h5py
import numpy as np
import pandas as pd
import torch
from scipy import io
from torch.utils.data.sampler import Sampler

NEW_DATASETS = ['Lung-Cancer', 'Movementlibras', 'Sonar']
NEW_DATASETS2 = ['waveform-5000']
NEW_DATASETS3 = ['UAV1', 'UAV2']
NEW_DATASETS6 = ['UJIndoorLoc']


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def encode2onehotarray(arr, num_classes):
    result = np.zeros([len(arr), num_classes])
    for i in range(len(arr)):
        result[i, arr[i]] = 1
    return result


def encode2onehot(arr):
    mapC = {}
    for i in range(len(arr)):
        if arr[i] not in mapC.keys():
            mapC[arr[i]] = 1
    x = arr[:, np.newaxis]
    one_hots = (np.array(list(mapC.keys())==x[:])).astype(np.integer)
    label = np.array([np.argmax(one_hot)for one_hot in one_hots])
    return label, one_hots.shape[-1]

def load_dataset(dataset):
    if dataset in NEW_DATASETS:
        mat = pd.read_csv('datasets/%s.csv' % dataset, header=None)
        X = mat.iloc[:,:-1]    # data
        X = X.astype(float)
        y = mat.iloc[:,-1]    # label
        X = np.array(X)
        y, NUM_CLASSES = encode2onehot(y)
    elif dataset in NEW_DATASETS2:
        mat = pd.read_csv('datasets/%s.csv' % dataset)
        X = mat.iloc[:,:-1]    # data
        X = X.astype(float)
        y = mat.iloc[:,-1]    # label
        X = np.array(X)
        y, NUM_CLASSES = encode2onehot(y)
    elif dataset in NEW_DATASETS3:
        arrays = {}
        f = h5py.File('datasets/%s.mat' % dataset, 'r')
        train_data = np.array(f['data_tr'])
        test_data = np.array(f['data_te'])
        X = np.concatenate((train_data,test_data),axis=1).T
        y = X[:, -1]
        X = X[:, :-1]
        y, NUM_CLASSES = encode2onehot(y)
    elif dataset in NEW_DATASETS6:
        mat = pd.read_csv('datasets/%s.csv' % dataset)
        X = mat.iloc[:,:-9]    # data
        X = X.astype(float)
        y = mat.iloc[:,-7]    # label
        X = np.array(X)
        y, NUM_CLASSES = encode2onehot(y)
    else:
        mat = io.loadmat('datasets/%s.mat' % dataset)
        X = mat['X']    # data
        X = X.astype(float)
        y = mat['Y']    # label
        y = y[:, 0]
        y_set = set()
        for i in y:
            y_set.add(i)
        NUM_CLASSES = len(y_set)
        if NUM_CLASSES > 2:
            y = y - 1
        else:
            for i in range(len(y)):
                if y[i] != 1:
                    y[i] = 0
    num_features = len(X[0])
    return X, y, NUM_CLASSES, num_features


class UnifLabelSampler(Sampler):
    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        nmb_non_empty_clusters = 0
        for i in range(len(self.images_lists)):
            if len(self.images_lists[i]) != 0:
                nmb_non_empty_clusters += 1

        size_per_pseudolabel = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for i in range(len(self.images_lists)):
            # skip empty clusters
            if len(self.images_lists[i]) == 0:
                continue
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res = np.concatenate((res, indexes))

        np.random.shuffle(res)
        res = list(res.astype('int'))
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)