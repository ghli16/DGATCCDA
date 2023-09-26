import torch
import torch.optim as optim
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data.dataloader import DataLoader
from torch.utils import data
# from torch.utils.data import DataLoader
import utils
# from data import dataset
# import data.dataset
# from models.model import MNN
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class MNN(nn.Module):
    def __init__(self, node_size, nhid0, nhid1, droput, alpha):
        super(MNN, self).__init__()
        self.encode0 = nn.Linear(node_size, nhid0)
        self.encode1 = nn.Linear(nhid0, nhid1)
        self.decode0 = nn.Linear(nhid1, nhid0)
        self.decode1 = nn.Linear(nhid0, node_size)
        self.droput = droput
        self.alpha = alpha

    def forward(self, adj_batch, adj_mat, b_mat):
        t0 = F.leaky_relu(self.encode0(adj_batch))
        t0 = F.leaky_relu(self.encode1(t0))
        embedding = t0
        t0 = F.leaky_relu(self.decode0(t0))
        t0 = F.leaky_relu(self.decode1(t0))
        embedding_norm = torch.sum(embedding * embedding, dim=1, keepdim=True)
        L_1st = torch.sum(adj_mat * (embedding_norm -
                                     2 * torch.mm(embedding, torch.transpose(embedding, dim0=0, dim1=1))
                                     + torch.transpose(embedding_norm, dim0=0, dim1=1)))
        L_2nd = torch.sum(((adj_batch - t0) * b_mat) * ((adj_batch - t0) * b_mat))
        return L_1st, self.alpha * L_2nd, L_1st + self.alpha * L_2nd

    def savector(self, adj):
        t0 = self.encode0(adj)
        t0 = self.encode1(t0)
        return t0


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', default='./datat/cora/cora_edgelist.txt',
                        help='Input graph file')
    parser.add_argument('--output', default='./datat/cora/Vec.emb',
                        help='Output representation file')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--weighted', action='store_true', default=False,
                        help='Treat graph as weighted')
    parser.add_argument('--epochs', default=100, type=int,
                        help='The training epochs of SDNE')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--alpha', default=1e-2, type=float,
                        help='alhpa is a hyperparameter in SDNE')
    parser.add_argument('--beta', default=5., type=float,
                        help='beta is a hyperparameter in SDNE')
    parser.add_argument('--nu1', default=1e-5, type=float,
                        help='nu1 is a hyperparameter in SDNE')
    parser.add_argument('--nu2', default=1e-4, type=float,
                        help='nu2 is a hyperparameter in SDNE')
    parser.add_argument('--bs', default=100, type=int,
                        help='batch size of SDNE')
    parser.add_argument('--nhid0', default=1000, type=int,
                        help='The first dim')
    parser.add_argument('--nhid1', default=128, type=int,
                        help='The second dim')
    parser.add_argument('--step_size', default=10, type=int,
                        help='The step size for lr')
    parser.add_argument('--gamma', default=0.9, type=int,
                        help='The gamma for lr')
    args = parser.parse_args()

    return args
class Dataload(data.Dataset):

    def __init__(self, Adj, Node):
        self.Adj = Adj
        self.Node = Node
    def __getitem__(self, index):
        return index
        # adj_batch = self.Adj[index]
        # adj_mat = adj_batch[index]
        # b_mat = torch.ones_like(adj_batch)
        # b_mat[adj_batch != 0] = self.Beta
        # return adj_batch, adj_mat, b_mat
    def __len__(self):
        return self.Node

def get_embedding_sdne(edg, Node):
    args = parse_args()
    edg1 = edg.cpu().t().detach().numpy()
    Adj = np.zeros([Node, Node], dtype=np.int32)
    for i in range(edg1.shape[0]):
        Adj[edg1[i][0], edg1[i][1]] = 1
        Adj[edg1[i][1], edg1[i][0]] = 1
    Adj = torch.FloatTensor(Adj)
    # G, Adj= dataset.Read_graph(edg1, Node)
    model = MNN(Node, args.nhid0, args.nhid1, args.dropout, args.alpha)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
    Data = Dataload(Adj, Node)
    Data = DataLoader(Data, batch_size=args.bs, shuffle=True, )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_sum, loss_L1, loss_L2, loss_reg = 0, 0, 0, 0
        for index in Data:
            adj_batch = Adj[index].cuda()
            adj_mat = adj_batch[:, index].cuda()
            b_mat = torch.ones_like(adj_batch).cuda()
            b_mat[adj_batch != 0] = args.beta

            opt.zero_grad()
            L_1st, L_2nd, L_all = model(adj_batch, adj_mat, b_mat)
            L_reg = 0
            for param in model.parameters():
                L_reg += args.nu1 * torch.sum(torch.abs(param)) + args.nu2 * torch.sum(param * param)
            Loss = L_all + L_reg
            Loss.backward()
            opt.step()
            loss_sum += Loss
            loss_L1 += L_1st
            loss_L2 += L_2nd
            loss_reg += L_reg
        scheduler.step(epoch)
        # print("The lr for epoch %d is %f" %(epoch, scheduler.get_lr()[0]))
        # print("loss for epoch %d is:" %epoch)
        # print("loss_sum is %f" %loss_sum)
        # print("loss_L1 is %f" %loss_L1)
        # print("loss_L2 is %f" %loss_L2)
        # print("loss_reg is %f" %loss_reg)
    model.eval()
    Adj = Adj.cuda()
    embedding = model.savector(Adj)
    outVec = embedding.cpu().detach().numpy()
    return outVec
    # np.savetxt(args.output, outVec)