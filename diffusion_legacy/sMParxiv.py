import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU, GELU, Dropout
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.utils import get_laplacian
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
# from logger import Logger
import argparse
import math

class UFGLevel(MessagePassing):
    def __init__(self, in_channels, dropout_prob=0.5, init_scale=1, LayerNorm=False):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        # self.linear = torch.nn.Linear(in_channels, out_channels)
        self.LayerNorm = LayerNorm
        self.filter = nn.Parameter(torch.Tensor(1, in_channels))
        nn.init.uniform_(self.filter, init_scale, init_scale+0.1)
        if self.LayerNorm:
            self.ln = nn.LayerNorm(in_channels)
        
    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [2, E], i.e. the ajacent matrix at one single level
        # Step 1: Linearly transform node feature matrix.
        # x = self.linear(x)
        if self.LayerNorm:
            return self.ln(self.propagate(edge_index, x=x, edge_attr=edge_attr) * self.filter)
        else:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr) * self.filter

    def message(self, x_j, edge_attr):
        # x_j has shape [E, out_channels]

        # Calculate the framelets coeff.     
        return edge_attr.view(-1, 1) * x_j
    
    

class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, d_list, dropout_prob=0.5, shortcut=False):
        super(Net, self).__init__()
        # self.GConv1 = UFGConv(num_features, nhid, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold)
        # self.GConv2 = UFGConv(nhid, num_classes, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold)
        # self.drop1 = nn.Dropout(dropout_prob)
        self.shortcut = shortcut
        self.edge_index_list, self.edge_attr_list = [], []
        for i in range(1, len(d_list)):
            self.register_buffer('edge_index_'+str(i), d_list[i].coalesce().indices())
            self.register_buffer('edge_attr_'+str(i), d_list[i].coalesce().values())
            self.edge_index_list.append(getattr(self, 'edge_index_' + str(i))) 
            self.edge_attr_list.append(getattr(self, 'edge_attr_' + str(i))) 
            

        self.ufg_list1 = nn.ModuleList([UFGLevel(nhid, init_scale=0.7*i) for i in range(1,len(d_list))])
        self.ufg_list2 = nn.ModuleList([UFGLevel(nhid, init_scale=0.7*i ) for i in range(1,len(d_list))])

        self.mlp1 = Seq(
                       ReLU(),
                       Dropout(dropout_prob),
                       Linear(3*nhid, num_classes)
                        )
        self.mlp2 = Seq(
                       ReLU(),
                       Dropout(dropout_prob),
                       Linear(3*nhid, nhid)
                        )
        self.mlp3 = Seq(
                       GELU(),
                       Dropout(dropout_prob),
                       Linear(nhid, num_classes)
                        )
        self.linear1 = Linear(num_features, nhid)
        self.linear = Linear(3*num_classes, num_classes)
        # self.linear1 = Linear(nhid, num_classes)
        # self.mlp2 = Seq(
        #                GELU(),
        #                Dropout(),
        #                Linear(3*out_channels, out_channels))
        self.linear2 = Linear(nhid, num_classes)

    def forward(self, x):
        # x has shape [num_nodes, num_input_features]
        x = self.linear1(x)
        if self.shortcut:
            shortcut = x

        x = torch.cat([ufg(x, edge_index, edge_attr) for 
                       edge_index, edge_attr, ufg in zip(self.edge_index_list, self.edge_attr_list, self.ufg_list1)], dim=1)
        # x = torch.cat([ufg(x, edge_index, edge_attr) for edge_index, edge_attr, ufg in self.combo1], dim=1)
        # shortcut = x
        # x = F.relu(shortcut + self.mlp(x))
        x = self.mlp2(x)  #+ self.mlp2(shortcut)
        # # x = self.linear1(self.mlp(x))
        x = torch.cat([ufg(x, edge_index, edge_attr) for 
                       edge_index, edge_attr, ufg in zip(self.edge_index_list, self.edge_attr_list, self.ufg_list2)], dim=1)
        x = self.mlp1(x)
        return F.log_softmax(x, dim=1)


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
def ChebyshevApprox(f, n):  # Assuming f : [0, pi] -> R.
    quad_points = 500
    c = [None] * n
    a = math.pi / 2
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



def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (UFGConv)')
    parser.add_argument('--log_steps', type=int, default=1,
                        help='the frequency of printing the results (default: 1)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of convolutional layers (default: 3')
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden units (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
    parser.add_argument('--bn', action='store_true',
                        help='apply batch normalization')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=5e-3,
                        help='weight decay (default: 5e-3)')
    parser.add_argument('--Lev', type=int, default=2,
                        help='level of transform (default: 2)')
    parser.add_argument('--s', type=float, default=2,
                        help='dilation scale > 1 (default: 2)')
    parser.add_argument('--n', type=int, default=2,
                        help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
    parser.add_argument('--FrameType', type=str, default='Haar',
                        help='frame type (default: Haar)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    args = parser.parse_args()
    print(args)

    # set random seed for reproducible results
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # determine the device cpu/gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # load dataset and further pre-process the data
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()

    # get the graph Laplacian L and save it as scipy sparse tensor
    num_nodes = data.adj_t.storage.sparse_sizes()[0]
    L = get_laplacian(torch.stack((data.adj_t.storage.row(), data.adj_t.storage.col())),
                      num_nodes=num_nodes, normalization='sym')
    # L_index = L[0].to(device)
    # L_value = L[1].to(device)
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

    lobpcg_init = np.random.rand(num_nodes, 1)
    lambda_max, _ = lobpcg(L, lobpcg_init, maxiter=50)
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
            d_list.append(scipy_to_torch_sparse(d[i, l]))

data = dataset[0].to(device)

dropout_prob=0.6
learning_rate=0.05
nhid=24
weight_decay=0.01

# get the node masks
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

# initialize the model
# model = Net(data.num_features, args.nhid, dataset.num_classes, r, Lev, num_nodes, shrinkage=None,
#             threshold=1e-3, dropout_prob=args.dropout, num_layers=args.num_layers, batch_norm=args.bn).to(device)
model = Net(data.num_features, nhid, dataset.num_classes, d_list, dropout_prob=0.6).to(device)
# initialize the evaluator and logger
evaluator = Evaluator(name='ogbn-arxiv')
# logger = Logger(args.runs, args)

# start training
for run in range(args.runs):
    print('****** Run {}: training start ******'.format(run + 1))
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, data, train_idx, optimizer)
        result = test(model, data, split_idx, evaluator)
        # logger.add_result(run, result)

        if epoch % args.log_steps == 0:
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}%, '
                  f'Test: {100 * test_acc:.2f}%')

#     logger.print_statistics(run)
# logger.print_statistics()