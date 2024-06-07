import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import argparse
import os.path as osp
from dataset_src.Graph_Dataset import *
import math
from torch_geometric.nn import GCNConv,GATConv,GINConv
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, GELU, ELU, Dropout
from torch_geometric.nn import global_add_pool
import matplotlib.pyplot as plt
# %% function for early_stopping
import time
from sklearn.metrics import roc_auc_score
from numba import jit

# %% FC Layer
def score_block(input_dim, output_dim, dropout):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(),
                         nn.Dropout(dropout))
# scattering transform
''' get all scattering coefficients '''
##################################construction of Scattering transform####################
class solo_pass(MessagePassing):
    def __init__(self, ):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

    def forward(self, scatter_edge_index, x,edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [2, E], i.e. the ajacent matrix at one single level
        x_out = self.propagate(scatter_edge_index, x=x, edge_attr=edge_attr)
        return x_out

    def message(self, x_j, edge_attr):
        # x_j has shape [edge_num, out_channels],all neighbour nodes
        # Calculate the framelets coeff. d_list[i]*f
        return edge_attr.view(-1, 1) * x_j

class Scatter(nn.Module):
    def __init__(self, in_channels, out_channels,dropout_prob=0.5, if_gat=True, head=2,layer=2):
        super(Scatter,self).__init__()  # "Add" aggregation (Step 5).
        self.dropout_prob = dropout_prob
        self.in_channels = in_channels
        self.gat = if_gat
        self.layer = layer
        self.solo_pass = solo_pass()#nn.ModuleList([solo_pass() for i in range(0, 7)])
        self.num_gat = int((3**layer-1)/2) # layer*(J-1) ####multiple times1+(3)+(9)*3=9,low pass1,2,4,8,16high pass2,4,8,16,32
        if self.gat==True:
            # print("using GAT!")
            self.conv_list = nn.ModuleList([GATConv(self.in_channels, out_channels, heads=head, dropout=dropout_prob) for i in range(0, self.num_gat)])
            self.mlp_list = nn.ModuleList([Seq(ELU(), Dropout(dropout_prob), Linear(head * out_channels, out_channels),nn.BatchNorm1d(out_channels)) for i in range(0, self.num_gat)])
        else:
            # print("with out using GAT!")
            self.mlp_learn = nn.ModuleList([Seq(ELU(), Dropout(dropout_prob), Linear(self.in_channels, out_channels),nn.BatchNorm1d(out_channels)) for i in range(0, self.num_gat)])
    def forward(self, x, scatter_edge_index, scatter_edge_attr, edge_index_o):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [2, E], i.e. the ajacent matrix at one single level
        x0 = self.solo_pass(scatter_edge_index[0], x=x, edge_attr=scatter_edge_attr[0])  ##low pass
        x_h1 = self.solo_pass(scatter_edge_index[1], x=x, edge_attr=scatter_edge_attr[1])
        x_h2 = self.solo_pass(scatter_edge_index[2], x=x, edge_attr=scatter_edge_attr[2])
        x_h3 = self.solo_pass(scatter_edge_index[3], x=x, edge_attr=scatter_edge_attr[3])
        x1 = self.solo_pass(scatter_edge_index[0], x=x_h1, edge_attr=scatter_edge_attr[0])
        x2 = self.solo_pass(scatter_edge_index[0], x=x_h2, edge_attr=scatter_edge_attr[0])
        x3 = self.solo_pass(scatter_edge_index[0], x=x_h3, edge_attr=scatter_edge_attr[0])
        if self.gat == True:
            x0 = self.mlp_list[0](self.conv_list[0](x0,edge_index_o))
            x1 = self.mlp_list[1](self.conv_list[1](x1,edge_index_o))
            x2 = self.mlp_list[2](self.conv_list[2](x2,edge_index_o))
            x3 = self.mlp_list[3](self.conv_list[3](x3, edge_index_o))
        else:
            x0 = self.mlp_learn[0](x0)
            x1 = self.mlp_learn[1](x1)
            x2 = self.mlp_learn[2](x2)
            x3 = self.mlp_learn[3](x3)

        x_out = x0+x1+x2+x3#torch.cat((x0,x1,x2,x3), dim=1)
        return x_out

class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, dropout_prob, hid_fc_dim, pRatio, head=4,layer=2,num_scatter=3,if_gat=True,if_linear=True):
        super(Net, self).__init__()
        self.dropout_prob = dropout_prob
        self.nhid = nhid
        self.pRatio = pRatio
        self.if_linear = if_linear
        self.num_classes=num_classes
        self.num_scatter = num_scatter
        if self.if_linear==True:
            # print("using linear transform")
            self.lin_in = Linear(num_features, nhid)
        else:
            # print("without using linear transform")
            nhid=num_features
        self.scatter_nn_list = nn.ModuleList([])
        for i in range(num_scatter):#######each scatt layer is two step of scattering say it is MS and here we use num_scatter MS form the Net
            self.scatter_nn_list.append(Scatter(in_channels=nhid, out_channels=nhid,dropout_prob=dropout_prob, if_gat=if_gat ,head=head,layer=layer))
        self.last_linear = nn.Linear(nhid, num_classes)
        # hidden_dim = [nhid] + hid_fc_dim + [num_classes]
        # fcList = [score_block(i, j, self.dropout_prob) for i, j in zip(hidden_dim[:-2], hidden_dim[1:])]
        # fcList.append(nn.Sequential(nn.Linear(*hidden_dim[-2:]),nn.BatchNorm1d(hidden_dim[-1])))
        # self.fc = nn.Sequential(*fcList)
    def forward(self, data, scatter_edge_index,scatter_edge_attr):
        input_x, edge_index = (data.x).float(), data.edge_index
        if self.if_linear == True:
            input_x = self.lin_in(input_x)
        hidden_rep = input_x
        for index in range(self.num_scatter):
            hidden_rep = hidden_rep + self.scatter_nn_list[index](hidden_rep,scatter_edge_index,scatter_edge_attr,edge_index) # not using += inplace operation,residule 
        x = self.last_linear(hidden_rep)
        # x = self.fc(hidden_rep)
        return F.log_softmax(x, dim=-1)
