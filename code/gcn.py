import os.path as osp

import argparse

from utils import *
from layers import GraphAttentionLayer, FeedForwardNetwork, LayerAtt
from SDNE1.sdmain import get_embedding_sdne

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv ,GATConv
import time
from deepwalkm.deepwalk.__main__ import get_embedding
import numpy as np
import random
# set_seed(666)

class Netsim(torch.nn.Module):
    def __init__(self, x, edge_index, hid_features, num_walk_emb, n_heads, end_embd, device, type):
        super(Netsim, self).__init__()

        self.conv1 = GraphAttentionLayer(x.shape[1], hid_features, n_heads,residual=True)
        self.conv2 = GraphAttentionLayer(hid_features, end_embd, n_heads, residual=True)

        self.conv1_deepwalk = GraphAttentionLayer(num_walk_emb, hid_features, n_heads, residual=True)
        self.conv2_deepwalk = GraphAttentionLayer(hid_features, end_embd, n_heads, residual=True)


        # 每个节点的所有边
        index = 0
        with open('gcn.adjlist', 'w') as f:
            for i in range(x.shape[0]):
                f.write(str(i))
                for j in range(index, edge_index.shape[1]):
                    if edge_index[0][j].item() == i:
                        f.write(' ' + str(edge_index[1][j].item()))
                    else:
                        index = j
                        break
                f.write('\n')

        # self.dat = get_embedding(edge_index, x.shape[0])
        edge_index1 = edge_index.cpu().detach().numpy()

        if type == "SDNE":

            self.data = get_embedding_sdne(edge_index, x.shape[0])
        else:
            # self.data = get_embedding_sdne(edge_index, x.shape[0])
            self.data = get_embedding(edge_index, x.shape[0], num_walk_emb)

        # self.data = get_embedding_sdne(edge_index, x.shape[0])
        # self.dat = torch.from_numpy(self.dat).float().cuda()
        self.dat = torch.from_numpy(self.data).float()

        tensor_index = torch.Tensor(5000, x.shape[0], num_walk_emb)
        tensor_neighbor_index = torch.Tensor(5000, x.shape[0], num_walk_emb)

        edgewight = []
        sim_list = []

        #  Updating adjacency matrix
        for index, row in enumerate(self.dat):
            row = torch.squeeze(row, 0)
            row = row.repeat(x.shape[0], 1)

            if index < 5000:
                tensor_index[index] = row
                tensor_neighbor_index[index] = self.dat
            else:
                if index % 5000 == 0:
                    sim = torch.cosine_similarity(tensor_index, tensor_neighbor_index, dim=-1)
                    sim_list.append(sim)
                tensor_index[index - 5000 * int(index / 5000)] = row
                tensor_neighbor_index[index - 5000 * int(index / 5000)] = self.dat

        if len(sim_list) <= 0:
            sim_ = torch.cosine_similarity(tensor_index, tensor_neighbor_index, dim=-1)
            sim = sim_[:x.shape[0]]
        else:
            sim = torch.cosine_similarity(tensor_index, tensor_neighbor_index, dim=-1)
            sim_list.append(sim)
            sim_ = torch.cat(sim_list, dim=0)
            sim = sim_[:x.shape[0]]

        index = 0
        adlist = []

        for i in range(x.shape[0]):
            lists = []
            for j in range(index, edge_index.shape[1]):
                if edge_index[0][j].item() == i:
                    lists.append(edge_index[1][j].item())
                else:
                    index = j
                    break
            adlist.append(lists)
        mask = torch.ones(sim.size()[0])
        mask = 1 - mask.diag()
        # cora 0.86
        # citeseer 0.9
        # pubmed 1
        sim_vec = torch.nonzero((sim > 0.92).float() * mask)
        for k in sim_vec:
            node_index = k[0].item()
            node_neighbor_index = k[1].item()
            if node_neighbor_index not in adlist[node_index]:
                adlist[node_index].append(node_neighbor_index)
        node_total = []
        neighbor_total = []
        for i in range(len(adlist)):
            for j in range(len(adlist[i])):
                node_total.append(i)
                neighbor_total.append(adlist[i][j])

        self.edge_index_new = torch.Tensor(2, len(node_total)).long()

        self.edge_index_new[0] = torch.from_numpy(np.array(node_total))
        self.edge_index_new[1] = torch.from_numpy(np.array(neighbor_total))
        self.edge_index_new = self.edge_index_new.to(device)
        self.device = device
        self.dropout = nn.Dropout(0.55)

        # 2层GAT：
        l = 2
        self.l = l
        self.cnn_y = nn.Conv2d(in_channels=l*2, out_channels=1,
                               kernel_size=(1, 1), stride=1, bias=True)
        self.fc1_x = nn.Linear(l*2, l * 5*2)
        self.fc2_x = nn.Linear(l * 5*2, l*2)
        self.sigmoidy = nn.Sigmoid()
        self.end_size = end_embd
        self.edge = edge_index

    #
    def forward(self, x):

        x_deepwalk = self.dat.float().to(self.device)

        x_1 = self.conv1(x, self.edge_index_new)
        x_deepwalk_1 = self.conv1_deepwalk(x_deepwalk, self.edge_index_new)


        x_2 = self.conv2(x_1, self.edge_index_new)
        x_deepwalk_2 = self.conv2_deepwalk(x_deepwalk_1, self.edge_index_new)

        xd = torch.cat((x_1, x_2, x_deepwalk_1, x_deepwalk_2), 1).t()
        # xd = torch.cat((x_1, x_2), 1).t()
        xd = xd.view(1, self.l*2, self.end_size, -1)

        globalAvgPool_y = nn.AvgPool2d((self.end_size, x.shape[0]), (1, 1))
        x_channel_att = globalAvgPool_y(xd)
        x_channel_att = x_channel_att.view(x_channel_att.size(0), -1)
        x_channel_att = self.fc1_x(x_channel_att)
        x_channel_att = self.fc2_x(x_channel_att)
        x_channel_att = self.sigmoidy(x_channel_att)
        x_channel_att = x_channel_att.view(x_channel_att.size(0), x_channel_att.size(1), 1 ,1)

        xD_channel_att = x_channel_att * xd
        xD_channel_att = torch.relu(xD_channel_att)

        y = self.cnn_y(xD_channel_att)
        y = y.view(self.end_size, x.shape[0]).t()
        #
        y = self.dropout(y)
        # y = self.att_cnn[0]*x_1 + self.att_cnn[1]*x_2 + self.att_cnn[2]*x_deepwalk_1 +self.att_cnn[3]*x_deepwalk_2
        return y


class Model(torch.nn.Module):
    def __init__(self, x, edge_index, sim1, edge_sim1, sim2, edge_sim2, hid_features, num_walk_emb, n_heads, end_embd, device):
        super(Model, self).__init__()
        self.m = Netsim(x,edge_index,hid_features,num_walk_emb ,n_heads, end_embd, device, "1")
        self.m1 = Netsim(sim1, edge_sim1, hid_features, num_walk_emb, n_heads, end_embd, device, "1")
        self.m2 = Netsim(sim2, edge_sim2, hid_features, num_walk_emb, n_heads, end_embd, device, "1")
        self.reconstructions = InnerProductDecoder1(sim1.shape[0], end_embd*2)
        self.att = Parameter(torch.rand(2), requires_grad=True)

    def forward(self,x, sim1, sim2, edge_idx_device):
        # torch.manual_seed(123)
        embd_x = self.m(x)
        embd_sim1 = self.m1(sim1)
        embd_sim2 = self.m2(sim2)
        embd_y = torch.cat((embd_sim1, embd_sim2), 0)
        # embd = torch.cat((embd_x, embd_y), 1)
        embd = self.att[0]*embd_x + self.att[1]*embd_y

        outputs = self.reconstructions(embd)
        return outputs











