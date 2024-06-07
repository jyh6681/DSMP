# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import GATConv
import math
import argparse
import os.path as osp

torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


# function for pre-processing
@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)

    return torch.sparse_coo_tensor(index, value, A.shape)


# function for pre-processing
def ChebyshevApprox(f, n):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)

    return c


# function for pre-processing
def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]

    return d


class UFGConv(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, shrinkage=None, threshold=1e-4, bias=True):
        super(UFGConv, self).__init__()
        self.Lev = Lev
        self.shrinkage = shrinkage
        self.threshold = threshold
        self.crop_len = (Lev - 1) * num_nodes
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).cuda())
#             self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1).cuda())
            self.filter1 = nn.Parameter(torch.Tensor(num_nodes, 1).cuda())
            self.filter2 = nn.Parameter(torch.Tensor(num_nodes, 1).cuda())
            self.filter3 = nn.Parameter(torch.Tensor(num_nodes, 1).cuda())
        else:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1))
        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(out_features).cuda())
            else:
                self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # GAT module
        self.GATconv1 = GATConv(out_features, out_features, heads=1, dropout=0.6)
        self.GATconv2 = GATConv(out_features, out_features, heads=1, dropout=0.6)
        self.GATconv3 = GATConv(out_features, out_features, heads=1, dropout=0.6)


    def reset_parameters(self):
#         nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.uniform_(self.filter1, 0.9, 1.1)
        nn.init.uniform_(self.filter2, 0.9, 1.1)
        nn.init.uniform_(self.filter3, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, d_list):
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        # x is a torch dense tensor
        x = torch.matmul(x, self.weight)

        # Fast Tight Frame Decomposition
        x1 = torch.sparse.mm(d_list[1], x)
        x2 = torch.sparse.mm(d_list[2], x)
        x3 = torch.sparse.mm(d_list[3], x)
#         x1 = torch.sparse.mm(torch.cat(d_list[:2], dim=0), x)
#         x2 = torch.sparse.mm(torch.cat(d_list[2:], dim=0), x)
#         x = torch.cat((x1, x2),0)
        # the output x has shape [r * Lev * num_nodes, #Features]

        # perform wavelet shrinkage (optional)
        if self.shrinkage is not None:
            if self.shrinkage == 'soft':
                x = torch.mul(torch.sign(x), (((torch.abs(x) - self.threshold) + torch.abs(torch.abs(x) - self.threshold)) / 2))
            elif self.shrinkage == 'hard':
                x = torch.mul(x, (torch.abs(x) > self.threshold))
            else:
                raise Exception('Shrinkage type is invalid')

        # Hadamard product in spectral domain
#         x = self.filter * x
        x1 = self.filter1 * x1
#         x2 = self.filter2 * x2
#         x3 = self.filter3 * x3
        # filter has shape [r * Lev * num_nodes, 1]
        # the output x has shape [r * Lev * num_nodes, #Features]
        
#         # GAT conv in spectral domain
#         x1 = F.dropout(F.elu(self.GATconv1(x1, edge_index)), p=0.6, training=self.training)
        x2 = F.dropout(F.elu(self.GATconv2(x2, edge_index)), p=0.6, training=self.training)
        x3 = F.dropout(F.elu(self.GATconv3(x3, edge_index)), p=0.6, training=self.training)

        # Fast Tight Frame Reconstruction
#         x = torch.sparse.mm(torch.cat(d_list[self.Lev - 1:], dim=0).transpose(0,1), x[self.crop_len:, :])
#         x1 = torch.sparse.mm(torch.cat(d_list[self.Lev - 1:2], dim=0).transpose(0,1), x1[self.crop_len:, :])
#         x2 = torch.sparse.mm(torch.cat(d_list[2:], dim=0).transpose(0,1), x2)
#         x = x1 + x2
        x = torch.sparse.mm(torch.cat(d_list[1:], dim=0).transpose(0,1), torch.cat((x1,x2,x3),0))
        if self.bias is not None:
            x += self.bias
        return x


class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, r, Lev, num_nodes, shrinkage=None, threshold=1e-4, dropout_prob=0.5):
        super(Net, self).__init__()
        self.GConv1 = UFGConv(num_features, nhid, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold)
        self.GConv2 = UFGConv(nhid, num_classes, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold)
        self.drop1 = nn.Dropout(dropout_prob)

    def forward(self, data, d_list):
        x = data.x  # x has shape [num_nodes, num_input_features]
        edge_index = data.edge_index
        x = F.relu(self.GConv1(x, edge_index, d_list))
        # x = self.drop1(x)
        # x = self.GConv2(x, edge_index, d_list)

        return F.log_softmax(x, dim=1)

torch.manual_seed(0)

# Training on CPU/GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# load dataset
dataname = 'Cora'
rootname = osp.join(osp.abspath(''), 'data', dataname)
dataset = Planetoid(root=rootname, name=dataname)

num_nodes = dataset[0].x.shape[0]
L = get_laplacian(dataset[0].edge_index, num_nodes=num_nodes, normalization='sym')
L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

lobpcg_init = np.random.rand(num_nodes, 1)
lambda_max, _ = lobpcg(L, lobpcg_init)
lambda_max = lambda_max[0]

## FrameType = 'Haar'
FrameType = 'Haar'
if FrameType == 'Haar':
    D1 = lambda x: np.cos(x / 2)
    D2 = lambda x: np.sin(x / 2)
    DFilters = [D1, D2]
    RFilters = [D1, D2]
elif FrameType == 'Linear':
    D1 = lambda x: np.square(np.cos(x / 2))
    D2 = lambda x: np.sin(x) / np.sqrt(2)
    D3 = lambda x: np.square(np.sin(x / 2))
    DFilters = [D1, D2, D3]
    RFilters = [D1, D2, D3]
elif FrameType == 'Quadratic':  # not accurate so far
    D1 = lambda x: np.cos(x / 2) ** 3
    D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
    D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
    D4 = lambda x: np.sin(x / 2) ** 3
    DFilters = [D1, D2, D3, D4]
    RFilters = [D1, D2, D3, D4]
else:
    raise Exception('Invalid FrameType')

Lev = 2  # level of transform
s = 2  # dilation scale
n = 2  # n - 1 = Degree of Chebyshev Polynomial Approximation
J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # dilation level to start the decomposition
r = len(DFilters)

# get matrix operators
d = get_operator(L, DFilters, n, s, J, Lev)
# enhance sparseness of the matrix operators (optional)
# d[np.abs(d) < 0.001] = 0.0
# store the matrix operators (torch sparse format) into a list: row-by-row
d_list = list()
for l in range(Lev):
    for i in range(r):
        d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))

# Hyper-parameter Settings
learning_rate = 0.01
weight_decay = 0.01
nhid = 16

# extract the data
data = dataset[0].to(device)

# create result matrices
num_epochs = 400
num_reps = 1
epoch_loss = dict()
epoch_acc = dict()
epoch_loss['train_mask'] = np.zeros((num_reps, num_epochs))
epoch_acc['train_mask'] = np.zeros((num_reps, num_epochs))
epoch_loss['val_mask'] = np.zeros((num_reps, num_epochs))
epoch_acc['val_mask'] = np.zeros((num_reps, num_epochs))
epoch_loss['test_mask'] = np.zeros((num_reps, num_epochs))
epoch_acc['test_mask'] = np.zeros((num_reps, num_epochs))
saved_model_val_acc = np.zeros(num_reps)
saved_model_test_acc = np.zeros(num_reps)

# initialize the learning rate scheduler
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# training
for rep in range(num_reps):
    print('****** Rep {}: training start ******'.format(rep + 1))
    max_acc = 0.0

    # initialize the model
    model = Net(dataset.num_node_features, nhid, dataset.num_classes, r, Lev, num_nodes, shrinkage=None,
                threshold=1e-3, dropout_prob=0.6).to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for epoch in range(num_epochs):
        # training mode
        model.train()
        optimizer.zero_grad()
        out = model(data, d_list)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # evaluation mode
        model.eval()
        out = model(data, d_list)
        for i, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = out[mask].max(dim=1)[1]
            correct = float(pred.eq(data.y[mask]).sum().item())
            e_acc = correct / mask.sum().item()
            epoch_acc[i][rep, epoch] = e_acc

            e_loss = F.nll_loss(out[mask], data.y[mask])
            epoch_loss[i][rep, epoch] = e_loss

            # scheduler.step(epoch_loss['val_mask'][rep, epoch])

            # print out results
            print('Epoch: {:3d}'.format(epoch + 1),
              'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
              'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
              'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
              'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
              'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
              'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            if epoch_acc['test_mask'][rep, epoch] > max_acc:
                max_acc = epoch_acc['test_mask'][rep, epoch]

    print('#### Rep {0:2d} Finished!  test acc: {1:.4f} ####\n'.format(rep + 1, max_acc))