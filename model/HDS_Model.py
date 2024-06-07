import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tnn
import math
from torch.nn.parameter import Parameter
from torch.nn import init
from torch_geometric.nn import global_add_pool

import json
from munch import munchify

config = {"J":3,"L":3,"sct_type":"diffusion","nhid":64,"dropout":0.1}
config = munchify(config)
# def load_config(path:str):
#   global config
#   with open(path) as file:
#     loader = json.load
#     config = loader(file)
#     config = munchify(config)
#     return config

# def set_param(params:dict):
#   global config
#   config = munchify(params)

def get_hyper():
    global config
    return config
# get_hyper()["J"], get_hyper()["L"], get_hyper()["sct_type"] = 3,3,"diffusion"
def backbone(name:str, in_features):
  if name.startswith("GCN"):
    return tnn.GCNConv(in_channels=in_features, out_channels= get_hyper().nhid)
  elif name.startswith("GAT"):
    return tnn.GATConv(in_channels=in_features, out_channels= get_hyper().nhid, heads=3, concat=False, dropout=get_hyper().dropout)
  elif name.startswith(("Sage", "SAGE")):
    return tnn.SAGEConv(in_channels=in_features, out_channels= get_hyper().nhid)
  elif name.startswith("GIN"):
    return tnn.GINConv(nn.Sequential(
        nn.Linear(in_features, get_hyper().nhid),
        nn.BatchNorm1d(get_hyper().nhid),
        nn.ReLU(inplace=True),
        nn.Linear(get_hyper().nhid, get_hyper().nhid),
    ))
  else:
    raise RuntimeError("unknown backbone")
class HDSGNN(nn.Module):
  def __init__(self, num_features, num_classes, gnn="GAT"):
    super(HDSGNN, self).__init__()
    self.num_features = num_features
    self.num_classes = num_classes
    self.layers = get_hyper().L
    self.orders = get_hyper().J
    self.dropout = get_hyper().dropout

    conv_list = []
    linear_list = []
    in_features = num_features
    for i in range(self.layers):
      conv_list.append(backbone(gnn,in_features))
      linear_list.append(nn.Linear(in_features=self.orders ** i * num_features, out_features= get_hyper().nhid))
      in_features = get_hyper().nhid * 2
    self.convs = nn.ModuleList(conv_list)
    self.lins = nn.ModuleList(linear_list)
    self.order_weights = Parameter(torch.Tensor(self.orders))
    self.conv_cls = tnn.GCNConv(in_channels=in_features, out_channels=get_hyper().nhid)
    self.global_add_pool = global_add_pool
    self.fc = nn.Linear(get_hyper().nhid,num_classes)
    self.bn = nn.BatchNorm1d(num_classes)
    self.reset_parameters()

  def reset_parameters(self):
    bound = 1 / math.sqrt(self.order_weights.size(0))
    init.uniform_(self.order_weights, -bound, bound)

  def forward(self, data):
    x, edge_index, feature = data.x, data.edge_index, data.diff_feat
    feature = torch.stack(feature,dim=0)
    # deal with special cae of PROTEINS
    if x.shape[-1]!=feature.shape[-1]:
        feature = feature[:,:,1:]
    batch = data.batch
    _, graph_size = torch.unique(batch, return_counts=True)
    begin_index = 0
    comb_feature = x
    for i in range(self.layers):
      convx = F.relu(self.convs[i](comb_feature, edge_index))
      convx = F.dropout(convx, self.dropout, self.training)

      count = self.orders ** i
      layer_feature = feature[begin_index: begin_index+count]# precomputed feature
      if i > 0:
        order_weight = self.order_weights.repeat(self.orders ** (i-1)).view(-1,1,1)
        layer_feature = order_weight * layer_feature
      layer_feature = torch.moveaxis(layer_feature,0,1)
      layer_feature = torch.reshape(layer_feature, (layer_feature.shape[0],-1))
      lin_sct = F.relu(self.lins[i](layer_feature))
      lin_sct = F.dropout(lin_sct, self.dropout, self.training)

      comb_feature = torch.cat([lin_sct, convx],dim=-1)
      begin_index += count
      
    convx = self.conv_cls(comb_feature, edge_index)
    self.global_add_pool
    h_pooled = self.global_add_pool(convx, batch)
    x = self.fc(h_pooled)
    if self.num_classes == 1:
        return x.view(-1)
    else:
        return x
    # return F.log_softmax(convx, dim=-1), convx
