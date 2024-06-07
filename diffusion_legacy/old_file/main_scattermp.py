import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import GATConv
import math, functools
import argparse
import os.path as osp
from typing import Optional
from torch_geometric.typing import OptTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU, GELU, ELU, Dropout
torch.set_default_dtype(torch.float)
torch.set_default_tensor_type(torch.FloatTensor)
# =============================================================================
# scattering transform
# =============================================================================
''' get one-step scattering coefficients using Paley-Littlewood wavelet '''
def propLayer(f, lamb, V, K=4, N=1):
    # K: scale; N: bandwidth
    idx = []
    for k in range(K):
        if k == 0:
            idx.append(lamb < N)
        else:
            idx.append((lamb >= 2 ** (k - 1) * N) * (lamb < 2 ** k * N))

    y = []
    for k in range(K):
        y.append(np.matmul(np.matmul(V[:, idx[k]], V[:, idx[k]].T), f))

    return y
''' get one-step scattering coefficients using a general wavelet '''
''' change the name of the function propLayerHaar to propLayer in order to use '''
''' using haar wavelet as an example, replace it with any wavelet '''

pi = math.pi#torch.from_numpy(np.array(np.pi).astype(np.float64))
def phi(lamb):
    lamb[lamb==0]=0.0000001
    y = 2*pi * lamb
    phi = torch.sin(y)/y
    return phi


def psi(lamb):
    lamb[lamb == 0] = 0.00000001
    y = pi * lamb
    psi = (torch.sin(lamb)/y) * (1 - torch.cos(lamb))
    return psi


def propLayerHaar(x, lamb, V, J=3):  # to replace propLayer,V is the adjacent matrix
    y = []
    for k in range(J):
        j = J - k
        if j == J:
            H = phi(2 ** j * lamb)
        else:
            H = psi(2 ** (-j) * lamb)
        H = torch.diag(H)
        y.append(torch.matmul(torch.matmul(torch.matmul(V, H), V.T), x))######wavelet transform
    return y


''' get all scattering coefficients '''

def getRep(x, lamb, V, layer=3):
    y_out = []
    y_next = []
    y = propLayerHaar(x, lamb, V)
    y_out.append(y.pop(0)) ##get popped item
    y_next.extend(y)
    for i in range(layer - 1):
        for k in range(len(y_next)):
            xtemp = y_next.pop(0)    ###get the popped item
            xtemp = torch.absolute(xtemp)#####abs
            y = propLayerHaar(xtemp, lamb, V)
            y_out.append(y.pop(0))
            y_next.extend(y)
    y_out = torch.cat(tuple(y_out), dim=1)  # use this to form a single matrix
    return y_out



##################################construction of Scattering transform####################
class Scatter(MessagePassing):
    def __init__(self, in_channels, out_channels, lamb,V,dropout_prob=0.5, if_mlp=True, head=2,layer=3):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

        self.if_mlp = if_mlp
        self.dropout_prob = dropout_prob

        self.act_drop = Seq(
            ELU(),
            Dropout(dropout_prob),
        )
        self.mlp = Seq(
            ELU(),
            Dropout(dropout_prob),
            Linear(head * out_channels, out_channels),
        )
        gcn_in = (pow(2,layer)-1)*in_channels
        self.conv = GATConv(gcn_in, out_channels, heads=head, dropout=dropout_prob)
        self.lamb=lamb
        self.V= V
        self.layer=layer

    def forward(self, x_in, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [2, E], i.e. the ajacent matrix at one single level
        x = getRep(x_in, self.lamb, self.V, layer=self.layer)
        if self.if_mlp:
            x = self.mlp(self.conv(self.propagate(edge_index, x=x), edge_index))
        else:
            x = self.act_drop(self.conv(self.propagate(edge_index, x=x), edge_index))

        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j has shape [E, out_channels]

        # Calculate the framelets coeff.
        return x_j#torch.abs(x_j)


class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, lamb,V, dropout_prob=0.5, shortcut=True, if_mlp=True, head=2):
        super(Net, self).__init__()
        self.shortcut = shortcut
        self.edge_index_list = []
        self.linear1 = Linear(num_features, nhid)
        self.scatter = Scatter(nhid, num_classes, lamb,V,dropout_prob=dropout_prob, if_mlp=if_mlp, head=head)

    def forward(self, x, edge_index):
        # x has shape [num_nodes, num_input_features]
        x = self.linear1(x)
        x = self.scatter(x, edge_index)
        return F.log_softmax(x, dim=1)


def main(args):
    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.nhid
    dropout_prob = args.dropout
    if_mlp = args.mlp
    head = args.head
    NormalizeFeatures = False
    dataname = args.dataname
    num_epochs = args.epoch
    num_reps = 5
    torch.manual_seed(0)

    # Training on CPU/GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load dataset

    rootname = osp.join('//', 'data', dataname)
    if NormalizeFeatures:
        dataset = Planetoid(root=rootname, name=dataname, transform=T.NormalizeFeatures())  #
    else:
        dataset = Planetoid(root=rootname, name=dataname)

    num_nodes = dataset[0].x.shape[0]
    L = get_laplacian(dataset[0].edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

    lamb, V = np.linalg.eigh(L.toarray())
    lamb = torch.from_numpy(lamb).to(device)
    V = torch.from_numpy(V).to(device)

    # extract the data
    data = dataset[0].to(device)
    data.x = data.x.float()
    # create result matrices

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

    # Hyper-parameter Settings

    # initialize the learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # training
    for rep in range(num_reps):
        # print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0

        # initialize the model
        model = Net(dataset.num_features, nhid, dataset.num_classes, lamb,V, dropout_prob=dropout_prob, if_mlp=if_mlp,
                    head=head).to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # evaluation mode
            model.eval()
            out = model(data.x, data.edge_index)
            for i, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = out[mask].max(dim=1)[1]
                correct = float(pred.eq(data.y[mask]).sum().item())
                e_acc = correct / mask.sum().item()
                epoch_acc[i][rep, epoch] = e_acc

                e_loss = F.nll_loss(out[mask], data.y[mask])
                epoch_loss[i][rep, epoch] = e_loss

                # scheduler.step(epoch_loss['val_mask'][rep, epoch])

                # print out results
            # print('Epoch: {:3d}'.format(epoch + 1),
            #   'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
            #   'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
            #   'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
            #   'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
            #   'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
            #   'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            if epoch_acc['val_mask'][rep, epoch] > max_acc:
                # torch.save(model.state_dict(), args.filename + '.pth')
                # print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                max_acc = epoch_acc['val_mask'][rep, epoch]
                record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        # print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc, record_test_acc))
    #    print('***************************************************************************************************************************')
    print('Average test accuracy over {0:2d} reps: {1:.4f} with stdev {2:.4f}'.format(num_reps,
                                                                                      np.mean(saved_model_test_acc),
                                                                                      np.std(saved_model_test_acc)))

    return np.mean(saved_model_test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("action", type=str, default=train, help="train or test")
    parser.add_argument("-e", "--epoch", help="Total epochs", type=int, default=100)
    parser.add_argument("-b", "--batch_size", help="Training batch size. Default is 100", default=10, type=int)
    parser.add_argument("-d", "--dataname", help="data name", default="Cora", type=str)
    parser.add_argument("-l", "--lr", help="Training learning rate. Default is 1e-4", type=float, default=0.0001)
    parser.add_argument("-w", "--wd", help="weight decay", type=float, default=0.01)
    parser.add_argument("-m", "--mlp", help="mlp", type=bool, default=True)
    parser.add_argument("-h", '--head', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--nhid', type=int, default=32)
    args = parser.parse_args()
    main(args)






