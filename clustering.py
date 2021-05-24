# Claim: Part of this file is from https://github.com/facebookresearch/deepcluster/blob/master/clustering.py

import time
import faiss
import numpy as np
import torch
import torch.utils.data as data
import warnings
warnings.filterwarnings("ignore", 'clustering')

USE_CUDA = False

__all__ = ['Kmeans', 'cluster_assign']


class ReassignedDataset(data.Dataset):
    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform
        self.dataset = dataset

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            # path = dataset[idx][0]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((idx, pseudolabel))
        return images

    def __getitem__(self, index):
        path, pseudolabel = self.imgs[index]
        img = self.dataset[path]
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)


def preprocess_features(npdata, pca=256):
    _, ndim = npdata.shape
    npdata =  np.ascontiguousarray(npdata.astype('float32'))

    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    npdata[np.isnan(npdata)] = 0.
    npdata[np.isinf(npdata)] = 0.

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    # for i in range(npdata.shape[0]):
    #     for j in range(npdata.shape[1]):
    #         if npdata[i][j] == np.inf:
    #             print("~~~!!!!!!!!", i, j)

    # npdata[np.isnan(npdata)] = 0.
    # npdata[np.isinf(npdata)] = 0.

    return npdata


def cluster_assign(images_lists, dataset):
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    return ReassignedDataset(image_indexes, pseudolabels, dataset, None)


def run_kmeans(x, nmb_clusters, verbose=False):
    n_data, d = x.shape
    clus = faiss.Clustering(d, nmb_clusters)
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    if USE_CUDA:
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)
    else:
        index = faiss.IndexFlatL2(d)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    stats = clus.iteration_stats
    losses = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]


class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        data_np = data.detach().cpu().numpy()
        pca = min(int(len(data_np)/2), 256)
        pca = min(pca, int(data_np.shape[1]/2))
        xb = preprocess_features(data_np, pca=pca)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data_np)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss
