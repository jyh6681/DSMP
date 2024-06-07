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

##################################construction of Scattering transform####################
class Scatter(MessagePassing):
    def __init__(self, in_channels, out_channels, lamb,V,dropout_prob=0.5, if_mlp=True, head=2,layer=3,J=3):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

        self.if_mlp = if_mlp
        self.dropout_prob = dropout_prob

        self.act_drop = Seq(
            ELU(),
            Dropout(dropout_prob),
        )
        self.J=J
        self.in_channels = in_channels
        num_gat = 2**layer-1#layer*(J-1) ####multiple times1+(3-1)+(3-1)*3=9,low pass1,2,4,8,16high pass2,4,8,16,32
        self.conv_list = nn.ModuleList([GATConv(in_channels, out_channels, heads=head, dropout=dropout_prob) for i in range(0,num_gat)])
        self.mlp_list = nn.ModuleList([Seq(nn.BatchNorm1d(head *out_channels),ELU(),Dropout(dropout_prob),Linear(head * out_channels, out_channels)) for i in range(0,num_gat)])
        self.lamb=lamb
        self.V= V
        self.layer=layer
        self.learn=True
        self.highpass=False

    def forward(self, x_in, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [2, E], i.e. the ajacent matrix at one single level
        #x = getRep(x_in, self.lamb, self.V, layer=self.layer)
        x_out = []
        x_next=[]
        x = propLayerHaar(x_in, self.lamb, self.V,self.J)
        if self.learn==True:
            x_out.append(self.mlp_list[0](self.conv_list[0](x.pop(0), edge_index)))
        else:
            x_out.append(x.pop(0))
        x_next.extend(x)
        if self.highpass==True:
            x_out.append(self.mlp_list[1](self.conv_list[0](x.pop(0), edge_index)))
            x_out.append(self.mlp_list[2](self.conv_list[0](x.pop(0), edge_index)))
        index = 1
        for i in range(1,self.layer):  ###layer-1
            for k in range(len(x_next)):    ###gat_num=3,6,12,len=2,4,8
                xtemp = torch.absolute(x_next.pop(0))  ###based on the last result to get new propagation, y_next has 2 before pop
                # if self.if_mlp:
                #     xtemp = self.mlp_list[index](self.conv_list[index](xtemp, edge_index)) ####coef with leanable parameter
                # else:
                #     xtemp = self.act_drop(self.conv_list[index](xtemp, edge_index))

                x = propLayerHaar(xtemp, self.lamb, self.V,self.J)
                if self.learn == True:
                    x_out.append(self.mlp_list[index](self.conv_list[index](x.pop(0), edge_index)))
                else:
                    x_out.append(x.pop(0))
                x_next.extend(x)
                index+=1
        x_out = torch.cat(tuple(x_out), dim=1)

        return x_out


class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, lamb,V, dropout_prob=0.5, shortcut=True, if_mlp=True, head=2,layer=3):
        super(Net, self).__init__()
        self.shortcut = shortcut
        self.linear1 = Linear(num_features, nhid)
        self.scatter1 = Scatter(num_features, nhid, lamb,V,dropout_prob=dropout_prob, if_mlp=if_mlp, head=head,layer=layer)
        self.mlp1 = Seq(
            nn.BatchNorm1d((2**layer-1)*nhid),
            ELU(),
            Dropout(dropout_prob),
            Linear((2**layer-1)*nhid, nhid),
        )
        # self.scatter2 = Scatter(nhid, nhid, lamb, V, dropout_prob=dropout_prob, if_mlp=if_mlp,head=head,layer=layer)
        # self.mlp2 = Seq(
        #     nn.BatchNorm1d((2 ** layer - 1) * nhid),
        #     ELU(),
        #     Dropout(dropout_prob),
        #     Linear((2 ** layer - 1) * nhid, nhid),
        # )
        self.mlp3 = Seq(
            nn.BatchNorm1d(nhid),
            ELU(),
            Dropout(dropout_prob),
            Linear(nhid, num_classes),
        )
    def forward(self, x, edge_index):
        # x has shape [num_nodes, num_input_features]
        x = self.mlp1(self.scatter1(x, edge_index))+ self.linear1(x) #####nhid
        #x = self.mlp2(self.scatter2(x, edge_index))+x  ###nhid
        #print("shape:",x.shape)
        x = self.mlp3(x)
        return F.log_softmax(x, dim=1)


def objective(learning_rate=0.01, weight_decay=0.01, nhid=64, dropout_prob=0.5, if_mlp=True, head=2,
              NormalizeFeatures=False, dataname='Cora',appdix="normal"):
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
    num_epochs = 100
    num_reps = 5
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
        train_loss = []
        val_loss = []
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            train_loss.append(loss.item())
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
    PATH = "/home/jyh_temp1/Downloads/scatter_MP/GFA-main/exp_result/scatter_2layer_acc"+str(round(np.mean(saved_model_test_acc),2))+\
           "std"+str(round(np.mean(saved_model_test_acc),2))+str(appdix)
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
        max_t=500,
        grace_period=100,
        reduction_factor=2,
        brackets=1)

    analysis = tune.run(
        training_function,
        config={
            "dataname": tune.grid_search(['Cora']),#, 'Cora', 'PubMed']),  # for ray tune, need abs path
            "learning_rate": tune.grid_search([5e-3]),
            "weight_decay": tune.grid_search([5e-3,1e-3]),

            "if_mlp": tune.grid_search([True]),
            "head": tune.grid_search([4]),
            "nhid": tune.grid_search([64]),
            "dropout_prob": tune.grid_search([0.5]),
            # "tqdm_disable": tune.grid_search(True),

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
    acc = objective(learning_rate=args.lr, weight_decay=args.wd, nhid=args.nhid, dropout_prob=args.drop, if_mlp=True, head=4,
              NormalizeFeatures=False, dataname='Cora',appdix=args.appdix)
    print(args,"accuracy:",acc)



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, default=ray_train, help="train or test")
    parse.add_argument("--lr", type=float, default=5e-3)
    parse.add_argument("-w", "--wd", type=float, default=5e-3)
    parse.add_argument("--drop", type=float, default=0.5)
    parse.add_argument("--nhid", type=int, default=64)
    parse.add_argument("-a", "--appdix", type=str, default='scatter', help="appendix")
    args = parse.parse_args()
    if args.action=="arg_train":
        arg_train(args)
    if args.action=="ray_train":
        ray_train()