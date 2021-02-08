import numpy as np
from model import SOFT
import os
import torch
import pandas as pd
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
from scipy import io
from skfeature.utility import construct_W
from sklearn.metrics.pairwise import pairwise_distances
import clustering
import argparse
from util import UnifLabelSampler, normalize, encode2onehotarray, load_dataset
import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NEW_DATASETS = ['Lung-Cancer', 'Movementlibras', 'Sonar']
NEW_DATASETS2 = ['waveform-5000']
NEW_DATASETS3 = ['UAV1', 'UAV2']
NEW_DATASETS6 = ['UJIndoorLoc']
DATASETS = ['COIL20', 'ORL', 'colon', 'madelon', 'Lung-Cancer', 'Movementlibras', 'Sonar', 'waveform-5000', 'UAV1', 'UAV2', 'UJIndoorLoc', 'nci9']
LEARNING_RATE = 1e-4
EPOCHS = 300
USE_CUDA = True


if not USE_CUDA:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--alpha', type=float, required=False, default=1.0)
parser.add_argument('-b', '--beta', type=float, required=False, default=0.001)
args = parser.parse_args()



def train(loader, model, crit, opt, epoch, num_classes):
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        target_attention = encode2onehotarray(target, num_classes)
        target_attention = torch.as_tensor(torch.from_numpy(target_attention), dtype=torch.float32)
        if USE_CUDA:
            target = target.cuda()
            input_tensor = input_tensor.cuda()
            target_attention = target_attention.cuda()
        input_var = torch.autograd.Variable(input_tensor)
        target_var = torch.autograd.Variable(target)
        target_attention_var = torch.autograd.Variable(target_attention)

        _, _, pseudo_label, _, _, _, pred_att, mask = model(X, A)

        loss_cluster_pred = crit(pseudo_label, target_var)
        loss_att = torch.mean( torch.sum(pred_att*target_attention_var, dim=1) )
        # loss_restrict = torch.abs((torch.sum(A_att) - torch.sum(A))) * 0.00
        loss_L21 = (torch.sum(torch.sqrt(torch.sum(mask ** 2, dim=1))) + torch.sum(torch.sqrt(torch.sum(mask ** 2, dim=0))))

        loss = loss_cluster_pred + loss_att * args.alpha + loss_L21 * args.beta

        # compute gradient and do SGD step
        opt.zero_grad()
        loss.backward()
        opt.step()

        # measure elapsed time
        end = time.time()

    return loss, loss_cluster_pred, loss_att, loss_L21






begin = time.time()
time_cost = []
for DATASET in DATASETS:
    X, y, NUM_CLASSES, num_features = load_dataset(DATASET)
    print(DATASET, "=============START!!", NUM_CLASSES, X.shape)

    model = SOFT(num_features, NUM_CLASSES, USE_CUDA=USE_CUDA)
    if USE_CUDA:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



    # kwargs = {"metric": "euclidean", "neighborMode": "knn", "weightMode": "heatKernel", "k": 10, 't': 1}
    # A = construct_W.construct_W(X.T, **kwargs).toarray()

    # A = np.ones([num_features, num_features])
    # A = A - A * np.eye(num_features)

    # A = np.random.randn(num_features, num_features) / 100 + 0.5
    # A = np.matmul(X.T, X)

    A = np.cov(X.T)
    A = np.abs(A)
    A = A - A * np.eye(num_features)
    A = normalize(A)
    A = A + np.eye(num_features)

    # A = pairwise_distances(X.T)
    # A = np.array(A)
    # A = np.max(A) - A
    # A = A - A * np.eye(num_features)


    start_time = time.time()
    X = torch.as_tensor(torch.from_numpy(X), dtype=torch.float32)   # N*F
    A = torch.as_tensor(torch.from_numpy(A), dtype=torch.float32)   # F*F
    y = torch.as_tensor(torch.from_numpy(y), dtype=torch.long)   # N*C

    deepcluster = clustering.Kmeans(NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()

    if USE_CUDA:
        X = X.cuda()
        A = A.cuda()
        y = y.cuda()
        criterion = criterion.cuda()

    for epoch in range(EPOCHS):
        flag, _, _, A_att, _, _, _, mask = model(X, A)

        clustering_loss = deepcluster.cluster(flag, verbose=0)
        train_dataset = clustering.cluster_assign(deepcluster.images_lists, X)

        # uniformly sample per target
        sampler = UnifLabelSampler(int(len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=len(train_dataset),
            sampler=sampler,
        )

        loss, loss_cluster_pred, loss_att, loss_L21 = train(train_dataloader, model, criterion, optimizer, epoch, NUM_CLASSES)


        if (epoch+1)%50 == 0 or epoch == 0:
            print('{:.0f} loss: {:.4f} loss_cluster_pred: {:.4f} loss_att: {:.4f} loss_L21: {:.4f} time:{:.4f}'.format(epoch+1, loss, loss_cluster_pred.data, loss_att.data, loss_L21.data, time.time()-start_time))
            sim_save = A_att.detach().cpu().numpy()
            sim_file = 'results/matrix/' + DATASET + '_' + str(epoch+1) + '_relation.csv'
            np.savetxt(sim_file, sim_save, delimiter = ',')

            mask_save = mask.detach().cpu().numpy()
            mask_file = 'results/matrix/' + DATASET + '_' + str(epoch+1) + '_mask.csv'
            np.savetxt(mask_file, mask_save, delimiter = ',')

    sim_save = A_att.detach().cpu().numpy()
    sim_file = 'results/' + DATASET + '_result_' + str(args.alpha) + '_' + str(args.beta) + '.csv'
    np.savetxt(sim_file, sim_save, delimiter = ',')

    mask_save = mask.detach().cpu().numpy()
    mask_file = 'results/' + DATASET + '_mask.csv'
    np.savetxt(mask_file, mask_save, delimiter = ',')

    time_cost.append(time.time()-start_time)

print("Total time:", time.time() - begin)
print(time_cost)