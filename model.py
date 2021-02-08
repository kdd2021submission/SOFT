import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class SOFT(nn.Module):
    def __init__(self, input_shape, nClass, USE_CUDA=True):
        super(SOFT, self).__init__()
        self.USE_CUDA = USE_CUDA
        self.input_shape = input_shape
        self.input_shape2 = int(input_shape / 2)
        self.nClass = nClass
        self.act = nn.ReLU()
        self.mask_weight = nn.Parameter(torch.from_numpy(np.random.normal(loc=0.0, scale=0.01, size=[input_shape, input_shape])).float() )
        self.pred_model = nn.Linear(self.input_shape, self.input_shape2)
        self.score_cl = nn.Linear(self.input_shape2, self.nClass)
        self.gcn1 = nn.Linear(self.input_shape, self.input_shape)
        self.gcn2 = nn.Linear(self.input_shape, self.input_shape)


    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1])
        if self.USE_CUDA:
            bn_module = bn_module.cuda()
        return bn_module(x)


    def forward(self, x, adj):
        # predict by original matrix
        adj_trans = adj
        c = torch.matmul(x, adj_trans)
        c = self.gcn1(c)
        c = self.act(c)
        c = self.apply_bn(c)
        c = torch.matmul(c, adj_trans)
        c = self.gcn2(c)
        c = self.act(c)
        c = self.apply_bn(c)
        pred = self.pred_model(c)
        pred = self.score_cl(pred)
        pred_cluster = pred
        pred = torch.softmax(pred, dim=1)
        pseudo_label = pred


        # create mask
        gcam = self.mask_weight + self.mask_weight.T
        mask = self.get_mask(gcam)
        masked_image = self.mask_image(adj_trans, mask)


        # predict by masked matrix
        c = torch.matmul(x, masked_image)
        c = self.gcn1(c)
        c = self.act(c)
        c = self.apply_bn(c)
        c = torch.matmul(c, masked_image)
        c = self.gcn2(c)
        c = self.act(c)
        c = self.apply_bn(c)
        masked_output = self.pred_model(c)
        masked_output = self.score_cl(masked_output)
        masked_output = F.sigmoid(masked_output)


        # get matrix after attention
        A_att = adj_trans - masked_image
        pred_att = masked_output


        # generate representations for psudo label by attentioned matrix
        c = torch.matmul(x, A_att)
        c = self.gcn1(c)
        c = self.act(c)
        c = self.apply_bn(c)
        c = torch.matmul(c, A_att)
        c = self.gcn2(c)
        c = self.act(c)
        c = self.apply_bn(c)
        flag = c

        return flag, pred_cluster, pseudo_label, A_att, pred, pred, pred_att, mask


    def get_mask(self, gcam, sigma=.5, w=8):
        mask = (gcam - torch.min(gcam)) / (torch.max(gcam) - torch.min(gcam))
        mask = F.sigmoid(w * (mask - sigma))
        return mask

    def mask_image(self, img, mask):
        return img - img * mask













class SOFT_FIRST(nn.Module):
    def __init__(self, input_shape, nClass, USE_CUDA=True):
        super(SOFT_FIRST, self).__init__()
        self.USE_CUDA = USE_CUDA
        self.input_shape = input_shape
        self.input_shape2 = int(input_shape / 2)
        self.nClass = nClass
        self.act = nn.ReLU()
        self.mask_weight = nn.Parameter(torch.from_numpy(np.random.normal(loc=0.0, scale=0.01, size=[input_shape, ])).float() )
        self.pred_model = nn.Linear(self.input_shape, self.input_shape2)
        self.score_cl = nn.Linear(self.input_shape2, self.nClass)
        self.gcn1 = nn.Linear(self.input_shape, self.input_shape)
        self.gcn2 = nn.Linear(self.input_shape, self.input_shape)



    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1])
        if self.USE_CUDA:
            bn_module = bn_module.cuda()
        return bn_module(x)


    def forward(self, x):

        c = self.gcn1(x)
        c = self.act(c)
        c = self.apply_bn(c)
        c = self.gcn2(c)
        c = self.act(c)
        c = self.apply_bn(c)
        pred = self.pred_model(c)
        pred = self.score_cl(pred)
        pred_cluster = pred
        pred = torch.softmax(pred, dim=1)
        pseudo_label = pred


        mask = self.get_mask(self.mask_weight)
        masked_image = self.mask_image(x, mask)


        c = self.gcn1(masked_image)
        c = self.act(c)
        c = self.apply_bn(c)
        c = self.gcn2(c)
        c = self.act(c)
        c = self.apply_bn(c)
        masked_output = self.pred_model(c)
        masked_output = self.score_cl(masked_output)
        masked_output = F.sigmoid(masked_output)


        A_att = 1 - mask
        pred_att = masked_output


        c = self.gcn1(x * mask)
        c = self.act(c)
        c = self.apply_bn(c)
        c = self.gcn2(c)
        c = self.act(c)
        c = self.apply_bn(c)
        flag = c

        return flag, pred_cluster, pseudo_label, A_att, pred, pred, pred_att, mask


    def get_mask(self, gcam, sigma=.5, w=8):
        mask = (gcam - torch.min(gcam)) / (torch.max(gcam) - torch.min(gcam))
        mask = F.sigmoid(w * (mask - sigma))
        return mask

    def mask_image(self, img, mask):
        return img - img * mask