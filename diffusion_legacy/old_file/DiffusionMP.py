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
from torch_geometric.utils import get_laplacian, degree
from scipy.sparse import csr_matrix

#L = get_laplacian(data.edge_index, num_nodes=num_nodes, normalization='sym')
class Scatter(MessagePassing):
    def __init__(self, in_channels, out_channels,dropout_prob=0.5, if_mlp=True, head=2,layer=3,J=3):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

        self.if_mlp = if_mlp
        self.dropout_prob = dropout_prob

        self.act_drop = Seq(
            ELU(),
            Dropout(dropout_prob),
        )
        self.J=J
        self.in_channels = in_channels
        num_gat = int((3**layer-1)/2)#layer*(J-1) ####multiple times1+(3-1)+(3-1)*3=9
        self.conv_list = nn.ModuleList([GATConv(in_channels, in_channels, heads=head, dropout=dropout_prob) for i in range(0,num_gat)])
        self.mlp_list = nn.ModuleList([Seq(ELU(),Dropout(dropout_prob),Linear(head * in_channels, out_channels)) for i in range(0,num_gat)])
        self.layer=layer

    def forward(self, x_in, edge_index,U,psi):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [2, E], i.e. the ajacent matrix at one single level

        f = x_in
        y_next = [f]
        low = torch.matmul(U, torch.absolute(f))
        y_out = [self.mlp_list[0](self.conv_list[0](low, edge_index))]
        index=1
        for i in range(self.layer - 1):
            for k in range(len(y_next)):
                y_next_new = []
                ftemp = y_next.pop(0)
                ftemp = torch.absolute(ftemp)
                y = [torch.matmul(torch.from_numpy(fltr).to(x_in.device).float(), ftemp) for fltr in psi]   ###psi x
                for y_temp in y:
                    coef=torch.matmul(U, torch.absolute(y_temp))
                    y_out.append(self.mlp_list[index](self.conv_list[index](coef, edge_index)))
                    index+=1
                y_next_new.extend(y)
            y_next = y_next_new
        y_out = torch.cat(tuple(y_out), dim=1)  # use this to form a single matrix

        return y_out

class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, dropout_prob=0.5, shortcut=True, if_mlp=True, head=2,layer=3):
        super(Net, self).__init__()
        self.shortcut = shortcut
        #self.scatter_list = nn.ModuleList([Scatter(in_channels, out_channels, lamb,V,dropout_prob=0.5, if_mlp=True, head=2,layer=3) for i in range(1,layer)])

        self.scatter = Scatter(num_features, nhid,dropout_prob=dropout_prob, if_mlp=if_mlp, head=head,layer=layer)
        self.mlp = Seq(
            ELU(),
            Dropout(dropout_prob),
            Linear(int((3**layer-1)/2)*nhid, num_classes),
        )

    def forward(self, x, edge_index,U,psi):
        # x has shape [num_nodes, num_input_features]
        x = self.scatter(x, edge_index,U,psi)
        x = self.mlp(x)
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
        dataset = Planetoid(root=rootname, name=dataname, transform= T.NormalizeFeatures())  #
    else:
        dataset = Planetoid(root=rootname, name=dataname)

    num_nodes = dataset[0].x.shape[0]

    # extract the data
    data = dataset[0].to(device)
    data.x = data.x.float()
    x_in = data.x
    edge_index=data.edge_index
    num_nodes = x_in.shape[0]
    np_edge = edge_index.cpu().numpy()
    row, col = np_edge[0], np_edge[1]
    adj_value = np.ones_like(row)
    W = csr_matrix((adj_value, (row, col)), shape=(num_nodes, num_nodes))
    D = degree(edge_index[0], num_nodes).cpu().numpy()
    Dhalf = np.diag(1 / np.sqrt(D))
    A = np.matmul(np.matmul(Dhalf, W.todense()), Dhalf)
    T = (np.eye(np.shape(D)[0]) + A) / 2#(np.eye(np.shape(D)[0])+np.matmul(W.todense(),np.diag(1/D)))/2#
    t = 3
    U = torch.from_numpy(np.linalg.matrix_power(T, t)).to(x_in.device)
    U = U.float()
    psi = []
    K = 3
    for idx in range(K):
        if idx == 0:
            psi.append(np.eye(np.shape(D)[0]) - T)
        else:
            T0 = T
            T = np.matmul(T0, T0)
            psi.append(T0 - T)
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
        model = Net(dataset.num_features, nhid, dataset.num_classes, dropout_prob=dropout_prob, if_mlp=if_mlp,
                    head=head).to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_loss = []
        val_loss = []
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index,U,psi)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            # evaluation mode
            model.eval()
            out = model(data.x, data.edge_index,U,psi)
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
    PATH = "/home/jyh_temp1/Downloads/scatter_MP/GFA-main/exp_result/diff_acc" + str(
        round(np.mean(saved_model_test_acc), 2)) + \
           "std" + str(round(np.mean(saved_model_test_acc), 2)) + str(appdix)
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
    ray.init(num_cpus=4, num_gpus=2)

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
            "learning_rate": tune.grid_search([0.005]),
            "weight_decay": tune.grid_search([5e-3,1e-3]),

            "if_mlp": tune.grid_search([True]),
            "head": tune.grid_search([4]),
            "nhid": tune.grid_search([64]),
            "dropout_prob": tune.grid_search([0.5,0.1]),
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



if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, default=ray_train, help="train or test")
    parse.add_argument("--lr", type=float, default=5e-3)
    parse.add_argument("-w", "--wd", type=float, default=5e-3)
    parse.add_argument("--drop", type=float, default=0.5)
    parse.add_argument("--nhid", type=int, default=64)
    parse.add_argument("-a", "--appdix", type=str, default='scatter', help="appendix")
    args = parse.parse_args()
    arg_train(args)
    ray_train()













