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
from logger import Logger
import argparse
import math


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


def GraphWFTG_Decomp(FD, L_index, L_value, c, s, J, Lev, num_nodes):
    # FD is a torch dense tensor
    # L is a torch sparse tensor
    r = len(c)
    n = len(c[0])
    a = math.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = FD
    d = torch.zeros(0, FD.shape[1]).to(FD.device)
    index = 0
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = spmm(L_index, ((s ** (-J + l - 1) / a) * L_value), num_nodes, num_nodes, T0F) - T0F
            d_temp = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = spmm(L_index, ((2 / a * s ** (-J + l - 1)) * L_value), num_nodes, num_nodes, T1F) - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d_temp += c[j][k] * TkF
            d = torch.cat((d, d_temp), dim=0)
        FD1 = d[index * num_nodes:(index + 1) * num_nodes, :]
        index += r

    return d


def GraphWFTG_Recon(d, L_index, L_value, c_rec, s, J, Lev, num_nodes):
    # this function is specific to FrameType == 'Haar'
    # d is a torch dense tensor
    # L is a torch sparse tensor
    r = len(c_rec)
    n = len(c_rec[0])
    a = math.pi / 2  # consider the domain of masks as [0, pi]
    block_index = list(np.arange(1, r * Lev, 2))
    block_index.append(block_index[-1] - 1)
    FD_recl = 0.0
    for l in np.arange(1, Lev + 1)[::-1]:
        for j in range(r):
            if (l == Lev) or (j > 0):
                index = block_index.pop()
                T0F = d[index * num_nodes:(index + 1) * num_nodes, :]
            else:
                T0F = FD_rec
            T1F = spmm(L_index, ((s ** (-J + l - 1) / a) * L_value), num_nodes, num_nodes, T0F) - T0F
            djl = (1 / 2) * c_rec[j][0] * T0F + c_rec[j][1] * T1F
            for k in range(2, n):
                TkF = spmm(L_index, ((2 / a * s ** (-J + l - 1)) * L_value), num_nodes, num_nodes, T1F) - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                djl += c_rec[j][k] * TkF
            FD_recl += djl
        FD_rec = FD_recl
        FD_recl = 0

    return FD_rec


class UFGConv(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, shrinkage=None, threshold=1e-4, bias=True):
        super(UFGConv, self).__init__()
        self.Lev = Lev
        self.num_nodes = num_nodes
        self.shrinkage = shrinkage
        self.threshold = threshold
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).cuda())
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1).cuda())
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

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, L_index, L_value, c, s, J):
        # x is a torch dense tensor
        x = torch.matmul(x, self.weight)

        # Fast Tight Frame Decomposition
        x = GraphWFTG_Decomp(x, L_index, L_value, c, s, J, self.Lev, self.num_nodes)
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
        x = self.filter * x
        # filter has shape [r * Lev * num_nodes, 1]
        # the output x has shape [r * Lev * num_nodes, #Features]

        # Fast Tight Frame Reconstruction
        x = GraphWFTG_Recon(x, L_index, L_value, c, s, J, self.Lev, self.num_nodes)

        if self.bias is not None:
            x += self.bias
        return x


class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, r, Lev, num_nodes, shrinkage=None, threshold=1e-4,
                 dropout_prob=0.5, num_layers=3, batch_norm=True):
        super(Net, self).__init__()

        self.batch_norm = batch_norm
        self.convs = torch.nn.ModuleList()
        self.convs.append(UFGConv(num_features, nhid, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold))
        if self.batch_norm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        for _ in range(num_layers - 2):
            self.convs.append(UFGConv(nhid, nhid, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold))
            if self.batch_norm:
                self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.convs.append(UFGConv(nhid, num_classes, r, Lev, num_nodes, shrinkage=shrinkage, threshold=threshold))

        self.dropout_prob = dropout_prob

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, L_index, L_value, c, s, J):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, L_index, L_value, c, s, J)
            if self.batch_norm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.convs[-1](x, L_index, L_value, c, s, J)
        return x.log_softmax(dim=-1)


def train(model, data, L_index, L_value, c, s, J, train_idx, optimizer, device):
    model.train()

    optimizer.zero_grad()
    out = model(data.x.to(device), L_index, L_value, c, s, J)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx].to(device))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, L_index, L_value, c, s, J, split_idx, evaluator, device):
    model.eval()

    out = model(data.x.to(device), L_index, L_value, c, s, J)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']].to(device),
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']].to(device),
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']].to(device),
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
    L_index = L[0].to(device)
    L_value = L[1].to(device)
    # L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

    # find the lambda_max of L
    # lobpcg_init = np.random.rand(num_nodes, 1)
    # lambda_max, _ = lobpcg(L, lobpcg_init)
    # lambda_max = lambda_max[0]
    lambda_max = 1.9782334
    print('lambda_max is: {0:.7f}'.format(lambda_max))

    # extract decomposition/reconstruction Masks
    FrameType = args.FrameType

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

    # specify the hyper-parameters for the framelet transforms
    Lev = args.Lev  # level of transform
    s = args.s  # dilation scale
    n = args.n  # n - 1 = Degree of Chebyshev Polynomial Approximation
    J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # dilation level to start the decomposition
    r = len(DFilters)

    # perform the Chebyshev Approximation
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)

    # get the node masks
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    # initialize the model
    model = Net(data.num_features, args.nhid, dataset.num_classes, r, Lev, num_nodes, shrinkage=None,
                threshold=1e-3, dropout_prob=args.dropout, num_layers=args.num_layers, batch_norm=args.bn).to(device)

    # initialize the evaluator and logger
    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    # start training
    for run in range(args.runs):
        print('****** Run {}: training start ******'.format(run + 1))
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, L_index, L_value, c, s, J, train_idx, optimizer, device)
            result = test(model, data, L_index, L_value, c, s, J, split_idx, evaluator, device)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()