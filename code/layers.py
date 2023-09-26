
import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot
from utils import *
# from SDNE.sdmain import get_embedding_sdne
from math import sqrt
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


class GraphAttentionLayer(MessagePassing):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 residual: bool, dropout: float = 0.7, slope: float = 0.2, activation: nn.Module = nn.ELU()):
        super(GraphAttentionLayer, self).__init__(aggr='mean', node_dim=0)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = n_heads
        self.residual = residual

        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope=slope)
        self.activation = activation

        self.feat_lin = Linear(in_features, out_features * n_heads, bias=True, weight_initializer='glorot')
        self.attn_vec = nn.Parameter(torch.Tensor(1, n_heads, out_features))

        # use 'residual' parameters to instantiate residual structure
        if residual:
            self.proj_r = Linear(in_features, out_features, bias=False, weight_initializer='glorot')
        else:
            self.register_parameter('proj_r', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.attn_vec)

        self.feat_lin.reset_parameters()
        if self.proj_r is not None:
            self.proj_r.reset_parameters()

    def forward(self, x, edge_idx, size=None):
        # normalize input feature matrix
        x = self.feat_dropout(x)

        x_r = x_l = self.feat_lin(x).view(-1, self.heads, self.out_features)

        # calculate normal transformer components Q, K, V
        output = self.propagate(edge_index=edge_idx, x=(x_l, x_r), size=size)

        if self.proj_r is not None:
            output = (output.transpose(0, 1) + self.proj_r(x)).transpose(1, 0)

        # output = self.activation(output)
        output = output.mean(dim=1)

        return output


# atten:

class LayerAtt(nn.Module):
    def __init__(self, inSize, outSize, gcnlayers):
        super(LayerAtt, self).__init__()
        self.layers = gcnlayers + 1
        self.inSize = inSize
        self.outSize = outSize
        self.q = nn.Linear(inSize, outSize)
        self.k = nn.Linear(inSize, outSize)
        self.v = nn.Linear(inSize, outSize)
        self.norm = 1 / sqrt(outSize)
        self.actfun1 = nn.Softmax(dim=1)
        self.actfun2 = nn.ReLU()
        self.attcnn = nn.Conv1d(in_channels=self.layers, out_channels=1, kernel_size=1, stride=1,
                            bias=True)

    def forward(self, x):# batchsize*gcn_layers*featuresize
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        out = torch.bmm(Q, K.permute(0, 2, 1)) * self.norm
        alpha = self.actfun1(out)# according to gcn_layers
        z = torch.bmm(alpha, V)
        # cnnz = self.actfun2(z)
        cnnz = self.attcnn(z)
        # cnnz = self.actfun2(cnnz)
        finalz = cnnz.squeeze(dim=1)

        return finalz

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, name, num_d, dropout=0., act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.name = name
        self.vars = {}
        self.issparse = False
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.num_d = num_d
        self.w = nn.Linear(input_dim * 2, input_dim * 2)
        self.w1 = nn.Linear(input_dim, input_dim)
        self.att_drug = nn.Parameter(torch.rand(2), requires_grad=True)
        self.att_cir = nn.Parameter(torch.rand(2), requires_grad=True)
        nn.init.xavier_uniform_(self.w1.weight)

    def forward(self, inputs, embd_cir, embd_drug):
        inputs = self.dropout(inputs)
        embd_drug = self.dropout(embd_drug)
        embd_cir = self.dropout(embd_cir)
        R = inputs[0:self.num_d, :]
        D = inputs[self.num_d:, :]
        R=torch.cat((R,embd_drug),1)
        D=torch.cat((D, embd_cir), 1)
        D = D.T
        x = R@D
        x = torch.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


class GTConv(nn.Module): # ajacency matrix weight in the meta-path
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))  #
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A

class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

