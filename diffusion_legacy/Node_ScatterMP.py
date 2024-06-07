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
import scipy.sparse as sp
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
    phi = np.sin(y)/y
    return phi


def psi(lamb):
    lamb[lamb == 0] = 0.00000001
    y = pi * lamb
    psi = (np.sin(lamb)/y) * (1 - np.cos(lamb))
    return psi


''' get all scattering coefficients '''

##################################construction of Scattering transform####################
def adj_torch_sparse(A):
    ed = sp.coo_matrix(A)
    indices = np.vstack((ed.row, ed.col))
    index = torch.Tensor(indices)
    values = torch.Tensor(ed.data)
    torch_sparse_mat = torch.sparse_coo_tensor(index, values, ed.shape)
    return torch_sparse_mat
def prop_scatter_haar(lamb, V, J=4):  # to replace propLayer,V is the adjacent matrix
    y = []
    for k in range(J):
        j = J - k
        if j == J:
            H = phi(2 ** j * lamb) ##2^3
        else:
            H = psi(2 ** (-j) * lamb)  ###2^-1,2^-2
        H = np.diag(H)
        scatter = np.matmul(np.matmul(V, H), V.T)
        #print("scatter:",scatter[0:10,0:10])
        if k==0:
            scatter[np.abs(scatter)<0.0005]=0
            print("low:",np.count_nonzero(scatter))
        else:
            scatter[np.abs(scatter) < 0.0001] = 0
            print("high:", np.count_nonzero(scatter))
        scatter = adj_torch_sparse(scatter).to("cuda")
        y.append(scatter)######wavelet transform
    return y

def prop_scatter_paley(lamb, V, K=4, N=1):
    # K: scale; N: bandwidth
    idx = []
    for k in range(K):
        if k == 0:
            idx.append( lamb < N )
        else:
            idx.append( (lamb >= 2**(k-1)*N) * (lamb < 2**k*N) )
    y = []
    for k in range(K):
        scatter = np.matmul(V[:,idx[k]], V[:,idx[k]].T)
        scatter = adj_torch_sparse(scatter).to("cuda")
        y.append( scatter)

    return y
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
        #print("pass::::::",edge_attr.shape)
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
            print("using GAT!")
            self.conv_list = nn.ModuleList([GATConv(self.in_channels, out_channels, heads=head, dropout=dropout_prob) for i in range(0, self.num_gat)])
            self.mlp_list = nn.ModuleList([Seq(ELU(), Dropout(dropout_prob), Linear(head * out_channels, out_channels),nn.BatchNorm1d(out_channels)) for i in range(0, self.num_gat)])
        else:
            print("with out using GAT!")
            self.mlp_learn = nn.ModuleList([Seq(ELU(), Dropout(dropout_prob), Linear(self.in_channels, out_channels),nn.BatchNorm1d(out_channels)) for i in range(0, self.num_gat)])
        self.mlp_all = Seq(nn.BatchNorm1d(self.num_gat*out_channels),ELU(),Dropout(dropout_prob),Linear(self.num_gat*out_channels, out_channels))
    def forward(self, x, scatter_edge_index, scatter_edge_attr, edge_index_o):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [2, E], i.e. the ajacent matrix at one single level
        x0 = self.solo_pass(scatter_edge_index[0], x=x, edge_attr=scatter_edge_attr[0])  ##low pass
        x_h1 = torch.abs(self.solo_pass(scatter_edge_index[1], x=x, edge_attr=scatter_edge_attr[1]))
        x_h2 = torch.abs(self.solo_pass(scatter_edge_index[2], x=x, edge_attr=scatter_edge_attr[2]))
        x_h3 = torch.abs(self.solo_pass(scatter_edge_index[3], x=x, edge_attr=scatter_edge_attr[3]))
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
    def __init__(self, num_features, nhid, num_classes, dropout_prob, head=4,layer=2,if_gat=True,if_linear=True):
        super(Net, self).__init__()
        self.dropout_prob = dropout_prob
        self.nhid = nhid
        self.if_linear = if_linear
        if self.if_linear:
            print("using linear transform")
            self.lin_in = Linear(num_features, nhid)
        else:
            print("without using linear transform")
            nhid=num_features
        self.scatter1 = Scatter(in_channels=nhid, out_channels=nhid,dropout_prob=dropout_prob, if_gat=if_gat ,head=head,layer=layer)
        self.scatter2 = Scatter(in_channels=nhid, out_channels=nhid, dropout_prob=dropout_prob,if_gat=if_gat , head=head,layer=layer)
        self.mlp = Seq(nn.BatchNorm1d(nhid),ELU(),Dropout(dropout_prob),Linear(nhid, num_classes))

    def forward(self, x, edge_index,scatter_list):
        self.scatter_edge_index_list, self.scatter_edge_attr_list = [], []
        for i in range(len(scatter_list)):
            self.scatter_edge_index_list.append(scatter_list[i].coalesce().indices())
            self.scatter_edge_attr_list.append(scatter_list[i].coalesce().values())
        if self.if_linear == True:
            x =self.lin_in(x)
        hidden_rep = x+ self.scatter1(x,self.scatter_edge_index_list,self.scatter_edge_attr_list,edge_index)
        hidden_rep = hidden_rep+ self.scatter2(hidden_rep,self.scatter_edge_index_list,self.scatter_edge_attr_list,edge_index)
        x = self.mlp(hidden_rep)
        return F.log_softmax(x, dim=1)

class SepNet(nn.Module):
    def __init__(self, num_features, nhid, num_classes, dropout_prob, head=4,layer=2,if_gat=True,if_linear=True):
        super(SepNet, self).__init__()
        self.dropout_prob = dropout_prob
        self.nhid = nhid
        self.gat = if_gat
        self.if_linear = if_linear
        if self.if_linear:
            print("using linear transform")
            self.lin_in = Linear(num_features, nhid)
        else:
            print("without using linear transform")
            nhid=num_features
        self.solo_pass = solo_pass()
        self.num_gat = int((3 ** layer - 1) / 2)+1  # layer*(J-1) ####multiple times1+(3)+(9)*3=9,low pass1,2,4,8,16high pass2,4,8,16,32
        if self.gat == True:
            print("using GAT!")
            self.conv_list = nn.ModuleList([GATConv(nhid, nhid, heads=head, dropout=dropout_prob) for i in range(0, self.num_gat)])
            self.mlp_list = nn.ModuleList([Seq(ELU(), Dropout(dropout_prob), Linear(head * nhid, nhid),nn.BatchNorm1d(nhid)) for i in range(0, self.num_gat)])
        else:
            print("with out using GAT!")
            self.mlp_learn = nn.ModuleList([Seq(ELU(), Dropout(dropout_prob), Linear(nhid, nhid), nn.BatchNorm1d(nhid)) for i in range(0, self.num_gat)])
        self.mlp = Seq(nn.BatchNorm1d(nhid),ELU(),Dropout(dropout_prob),Linear(nhid, num_classes),)

    def forward(self, x, edge_index, scatter_list):
        scatter_edge_index, scatter_edge_attr = [], []
        for i in range(len(scatter_list)):
            scatter_edge_index.append(scatter_list[i].coalesce().indices())
            scatter_edge_attr.append(scatter_list[i].coalesce().values())
        if self.if_linear:
            x = self.lin_in(x)
        x0 = self.solo_pass(scatter_edge_index[0], x=x, edge_attr=scatter_edge_attr[0])  ##low pass
        x_h1 = torch.abs(self.solo_pass(scatter_edge_index[1], x=x, edge_attr=scatter_edge_attr[1]))
        x_h2 = torch.abs(self.solo_pass(scatter_edge_index[2], x=x, edge_attr=scatter_edge_attr[2]))
        x_h3 = torch.abs(self.solo_pass(scatter_edge_index[3], x=x, edge_attr=scatter_edge_attr[3]))
        x_h4 = torch.abs(self.solo_pass(scatter_edge_index[4], x=x, edge_attr=scatter_edge_attr[4]))
        x1 = self.solo_pass(scatter_edge_index[0], x=x_h1, edge_attr=scatter_edge_attr[0])
        x2 = self.solo_pass(scatter_edge_index[0], x=x_h2, edge_attr=scatter_edge_attr[0])
        x3 = self.solo_pass(scatter_edge_index[0], x=x_h3, edge_attr=scatter_edge_attr[0])
        x4 = self.solo_pass(scatter_edge_index[0], x=x_h4, edge_attr=scatter_edge_attr[0])
        if self.gat == True:
            x0 = self.mlp_list[0](self.conv_list[0](x0, edge_index))
            x1 = self.mlp_list[1](self.conv_list[1](x1, edge_index))
            x2 = self.mlp_list[2](self.conv_list[2](x2, edge_index))
            x3 = self.mlp_list[3](self.conv_list[3](x3, edge_index))
            x4 = self.mlp_list[4](self.conv_list[4](x4, edge_index))
        else:
            x0 = self.mlp_learn[0](x0)
            x1 = self.mlp_learn[1](x1)
            x2 = self.mlp_learn[2](x2)
            x3 = self.mlp_learn[3](x3)
            x4 = self.mlp_learn[4](x4)
        x_stage1 = x + x0
        x_stage2 = x_stage1 + x1+x2+x3+x4
        x = self.mlp(x_stage2)
        return F.log_softmax(x, dim=1)

    def adjConcat(self, a, b):
        '''
        concat two adj matrix along the diag
        '''
        lena = len(a)
        lenb = len(b)
        left = np.row_stack((a, np.zeros((lenb, lena))))  # get the left
        right = np.row_stack((np.zeros((lena, lenb)), b))  # get the right
        result = np.hstack((left, right))
        return result
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

def objective(learning_rate=0.01, weight_decay=0.01, nhid=64, dropout_prob=0.5, if_gat=True, if_linear=True,head=4,
              NormalizeFeatures=False, dataname='Cora',appdix="normal",wavelet="haar"):
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
    import time
    start = time.time()
    num_nodes = dataset[0].x.shape[0]
    L = get_laplacian(dataset[0].edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))
    lamb, V = np.linalg.eigh(L.toarray())
    ###get the scatter list
    if wavelet=="haar":
        scatter_list = prop_scatter_haar(lamb,V)
    else:
        ##using discret wavelet
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
        d_list.append(scipy_to_torch_sparse(sparse.identity(L.shape[0])).to(device))
        for l in range(Lev):
            for i in range(r):
                d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))
        scatter_list=d_list

    # extract the data
    data = dataset[0].to(device)
    data.x = data.x.float()
    # create result matrices
    num_epochs = 100
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

    # Hyper-parameter Settings

    # initialize the learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # training
    for rep in range(num_reps):
        # print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0

        # initialize the model
        model = SepNet(dataset.num_features, nhid, dataset.num_classes, dropout_prob, head=head,layer=2,if_gat=if_gat,if_linear=if_linear).to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_loss = []
        val_loss = []
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index,scatter_list)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            # evaluation mode
            model.eval()
            out = model(data.x, data.edge_index,scatter_list)
            for i, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = out[mask].max(dim=1)[1]
                correct = float(pred.eq(data.y[mask]).sum().item())
                e_acc = correct / mask.sum().item()
                epoch_acc[i][rep, epoch] = e_acc

                e_loss = F.nll_loss(out[mask], data.y[mask])
                epoch_loss[i][rep, epoch] = e_loss
                if i=="val_mask":
                    val_loss.append(e_loss.item())

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
    import matplotlib.pyplot as plt
    PATH = "/home/jyh_temp1/Downloads/scatter_MP/GFA-main/node_class_result/"+str(wavelet)+str(dataname)+"acc"+str(round(np.mean(saved_model_test_acc),2))+\
           "std"+str(round(np.mean(saved_model_test_acc),2)) +"if_gat"+str(if_gat)+"if_linear"+str(if_linear)+ str(appdix)+'.pth'
    epoch_list = list(range(0,num_epochs))
    ax = plt.gca()
    plt.plot(epoch_list, train_loss, 'r-', label="train-loss")
    plt.plot(epoch_list, val_loss, 'b-', label="val-loss")
    plt.title('Training loss&Validation loss vs. epoches')
    plt.xlabel("epoches")
    plt.ylabel('Training loss&Val loss')
    plt.legend()
    plt.savefig(PATH +str(dataname) + '_curve.png', bbox_inches='tight')
    return np.mean(saved_model_test_acc)


def ray_train():
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler

    class ExperimentTerminationReporter(CLIReporter):
        def should_report(self, trials, done=True):
            """Reports only on experiment termination."""
            return done

    def training_function(config):
        # Hyperparameters
        # modes, width, learning_rate = config["modes"], config["width"], config["learning_rate"]
        #     for step in range(10):
        # Iterative training function - can be any arbitrary training procedure.
        # intermediate_score = fourier_2d_hopt_train.objective(modes, width, learning_rate)
        # Feed the score back back to Tune.
        acc = objective(**config)
        tune.report(acc=acc)

    ray.shutdown()
    ray.init(num_cpus=8, num_gpus=4)

    asha_scheduler = ASHAScheduler(
        # time_attr='training_iteration',
        metric='loss',
        mode='min',
        max_t=200,
        grace_period=100,
        reduction_factor=2,
        brackets=1)

    analysis = tune.run(
        training_function,
        config={
            "dataname": tune.grid_search(['Cora']),#, 'Cora', 'PubMed']),  # for ray tune, need abs path
            "learning_rate": tune.grid_search([5e-3]),
            "weight_decay": tune.grid_search([1e-4]),

            "if_gat": tune.grid_search([True,False]),
            "if_linear": tune.grid_search([False]),
            "head": tune.grid_search([4,8]),
            "nhid": tune.grid_search([64,128]),
            "dropout_prob": tune.grid_search([0.2,0.1]),
            "wavelet": tune.grid_search(["cheby"]),

        },
        progress_reporter=ExperimentTerminationReporter(),
        resources_per_trial={'gpu': 1, 'cpu': 2},
        # scheduler=asha_scheduler
    )

    print("Best config: ", analysis.get_best_config(
        metric="acc", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
def arg_train(args):
    acc = objective(learning_rate=args.lr, weight_decay=args.wd, nhid=args.nhid, dropout_prob=args.drop, head=4,if_linear=args.if_linear,if_gat=args.if_gat,
              NormalizeFeatures=False, dataname=args.dataset,appdix=args.appdix)
    print(args,"accuracy:",acc)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default=ray_train, help="ray or arg train")
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='name of dataset (default: Cora)')
    parser.add_argument('--reps', type=int, default=1,
                        help='number of repetitions (default: 10)')
    parser.add_argument("-b", '--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument("-e", '--epochs', type=int, default=100,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--patience', type=int, default=20,
                        help='early stopping criteria (default: 20)')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=1e-3,
                        help='weight decay (default: 5e-3)')
    parser.add_argument('--nhid', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--num_conv_layer', type=int, default=2,
                        help='number of hidden mlp layers in a conv layer (default: 2)')
    parser.add_argument('--drop', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--appdix', type=str, default='results',
                        help='filename to store results and the model (default: results)')
    parser.add_argument('--if_gat', type=str, default=True)
    parser.add_argument('--if_linear', type=str, default=True)
    parser.add_argument('--scattertype', type=str, default='haar')
    args = parser.parse_args()
    if args.action=="arg_train":
        arg_train(args)
    if args.action=="ray_train":
        ray_train()






