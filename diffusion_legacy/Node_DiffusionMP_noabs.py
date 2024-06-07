import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from new_dataset import HetroDataSet
from torch_geometric.datasets import Planetoid, WikiCS,Coauthor
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


##################################construction of Scattering transform####################
def adj_torch_sparse(A):
    ed = sp.coo_matrix(A)
    indices = np.vstack((ed.row, ed.col))
    index = torch.LongTensor(indices)
    values = torch.Tensor(ed.data)
    torch_sparse_mat = torch.sparse_coo_tensor(index, values, ed.shape)
    return torch_sparse_mat

def prop_scatter_diffusion(edge_index, K=3,diffusion="sym"):
    y_out=[]
    num_nodes = edge_index.max().item() + 1
    np_edge = edge_index.cpu().numpy()
    row, col = np_edge[0], np_edge[1]
    adj_value = np.ones_like(row)
    W = csr_matrix((adj_value, (row, col)), shape=(num_nodes, num_nodes))
    D = degree(edge_index[0], num_nodes).cpu().numpy()
    if np.min(D)==0:
        D = np.where(D>0,D,1)
    Dhalf = np.diag(1 / np.sqrt(D))
    A = np.matmul(np.matmul(Dhalf, W.todense()), Dhalf)
    if diffusion=="sym":
        T = (np.eye(np.shape(D)[0]) + A) / 2 ###(I+T)/2,sym diffusion
    else:
        T = (np.eye(np.shape(D)[0]) + np.matmul(W.todense(), np.diag(1 / D))) / 2
    t = 2^(K-1)
    U = np.linalg.matrix_power(T, t)  ###U=T^3(2^2-1=3)
    # U = np.eye(num_nodes)
    U = adj_torch_sparse(U).to("cuda")
    y_out.append(U)
    psi = []
    for idx in range(K):
        if idx == 0:
            tmp=np.eye(np.shape(D)[0]) - T
            tmp = adj_torch_sparse(tmp).to("cuda")
            psi.append(tmp) ###I-T
        else:
            T0 = T
            T = np.matmul(T0, T0)
            tmp = T0-T
            tmp=adj_torch_sparse(tmp).to("cuda")
            psi.append(tmp)###T^(2^(j-1))-T^(2^j)
    ##psi={psi_0,psi_1,psi_2}
    y_out.extend(psi)
    return y_out
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
            print("using GAT!")
            self.conv_list = nn.ModuleList([GATConv(self.in_channels, out_channels, heads=head, dropout=dropout_prob) for i in range(0, self.num_gat)])
            self.mlp_list = nn.ModuleList([Seq(ELU(), Dropout(dropout_prob), Linear(head * out_channels, out_channels),nn.BatchNorm1d(out_channels)) for i in range(0, self.num_gat)])
        else:
            print("with out using GAT!")
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
    def __init__(self, num_features, nhid, num_classes, dropout_prob,head=4,layer=2,num_scatter=3,if_gat=True,if_linear=True):
        super(Net, self).__init__()
        self.dropout_prob = dropout_prob
        self.nhid = nhid
        self.if_linear = if_linear
        self.num_classes=num_classes
        self.num_scatter = num_scatter
        if self.if_linear==True:
            print("using linear transform")
            self.lin_in = Linear(num_features, nhid)
        else:
            print("without using linear transform")
            nhid=num_features
        self.scatter_list = nn.ModuleList([])
        for i in range(num_scatter):
            self.scatter_list.append(Scatter(in_channels=nhid, out_channels=nhid,dropout_prob=dropout_prob,if_gat=if_gat ,head=head,layer=layer))
        self.mlp = Seq(nn.BatchNorm1d(nhid),ELU(),Dropout(dropout_prob),Linear(nhid, num_classes),)

    def forward(self, x, edge_index,scatter_list):
        self.scatter_edge_index_list, self.scatter_edge_attr_list = [], []
        for i in range(len(scatter_list)):
            self.scatter_edge_index_list.append(scatter_list[i].coalesce().indices())
            self.scatter_edge_attr_list.append(scatter_list[i].coalesce().values())
        if self.if_linear == True:
            x = self.lin_in(x)
        hidden_rep = x
        for index in range(self.num_scatter):
            hidden_rep += self.scatter_list[index](hidden_rep, self.scatter_edge_index_list,self.scatter_edge_attr_list,edge_index)
        out = self.mlp(hidden_rep)
        # data = data.to(torch.float32).cuda()
        # structure = structure.cuda()
        # x = self.GConv[0](data,structure)
        # if self.conv_type.lower() == 'gat':
        #     x = F.elu(x)
        # elif self.conv_type.lower() == ('gcn' or 'ufg_s'):
        #     x = F.relu(x)
        # x = self.drop1(x)
        # x = self.GConv[1](x, structure)
        # return F.log_softmax(x, dim=-1)
        return F.log_softmax(out, dim=1)


def compress_feat(feat):
    ####get maximum feat size, non zeros number in each row###
    row,col = feat.shape
    max = 0
    for i in range(row):
        num = np.count_nonzero(feat[i])
        if num>max:
            max=num
    new_feat = np.zeros((row,max))
    ###set value for new feat
    for i in range(row):
        nonzero = np.nonzero(feat[i])[0]
        for j in range(nonzero.shape[0]):
            #new_feat[i,j]= feat[i,nonzero[j]]
            new_feat[i, j] = np.sin(nonzero[j]/col)
    return new_feat


def objective(learning_rate=0.01, weight_decay=0.01, nhid=64, dropout_prob=0.5, num_epochs = 200,num_scatter=3,if_gat=True, if_linear=False,head=4,
              NormalizeFeatures=True, dataname='Cora',appdix="normal",diffusion="sym"):
    torch.manual_seed(0)

    # Training on CPU/GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    rootname = osp.join("/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/", 'data', dataname)
    # load dataset
    if dataname.lower() == 'wikics':
        print("load wikics")
        dataset = WikiCS(root=rootname)
        data = dataset[0].to(device)
        data.x = data.x.float()
    if dataname.lower() == 'cs':
        dataset = Coauthor(root=rootname, name=dataname)
        num_class = dataset.num_classes
        train_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        train_mask[0:300] = 1
        val_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        val_mask[300:500] = 1
        test_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        test_mask[500:1500] = 1
        data = Data(x=dataset[0].x, edge_index=dataset[0].edge_index, y=dataset[0].y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask).to(device)
        data.x = data.x.float()
    if dataname.lower() == 'cora' or dataname.lower() == 'pubmed' or dataname.lower() == 'citeseer':
        print("load Planet data")
        dataset = Planetoid(root=rootname, name=dataname)
        data = dataset[0].to(device)
        data.x = data.x.float()
        # print("line graph:",data.edge_attr,cora_linegraph,"\n",reverse_cora)#,cora_linegraph.x.shape,cora_linegraph.edge_index.shape,cora_linegraph.edge_atrr)
    if dataname.lower() == 'wisconsin' or dataname.lower() == 'texas':
        rootname = osp.join("/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/", 'data')
        dataset = HetroDataSet(root=rootname, name=dataname)
        data = dataset[0].to(device)
        data.x = data.x.float()
    num_features = dataset.num_features
    ####compress feature matrix
    # new_feat = compress_feat(data.x.cpu().numpy())
    # data.x = torch.from_numpy(new_feat).to(device).float()
    # num_features = new_feat.shape[1]
    ###prepare scatteringoperator
    edge_index=dataset[0].edge_index
    ###get the scatter list
    scatter_list = prop_scatter_diffusion(edge_index,K=3,diffusion=diffusion)
    # create result matrices
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

    # training
    for rep in range(num_reps):
        # print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0

        # initialize the model
        model = Net(num_features, nhid, dataset.num_classes, dropout_prob, head=head,layer=2,num_scatter=num_scatter,if_gat=if_gat,if_linear=if_linear).to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # initialize the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.96, patience=3,verbose=False,min_lr=1e-6)
        train_loss = []
        val_loss = []
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            data = data.to(device)
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

                scheduler.step(epoch_loss['val_mask'][rep, epoch])

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
    print("args: "+str(dataname)+"if_gat"+str(if_gat)+"if_linear"+str(if_linear)+"lr"+str(learning_rate)+"wd"+str(weight_decay))
    print('Average test accuracy over {0:2d} reps: {1:.4f} with stdev {2:.4f}'.format(num_reps,
                                                                                      np.mean(saved_model_test_acc),
                                                                                      np.std(saved_model_test_acc)))
    import matplotlib.pyplot as plt
    PATH = "/home/jyh_temp1/Downloads/scatter_MP/GFA-main/node_class_result/diffuion_"+str(diffusion)+str(dataname)+"acc"+str(round(np.mean(saved_model_test_acc),2))+\
           "std"+str(round(np.mean(saved_model_test_acc),2)) +"num_scatter"+str(num_scatter)+"if_gat"+str(if_gat)+"if_linear"+str(if_linear)+ str(appdix)+'.pth'
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
    ray.init(num_cpus=8, num_gpus=2)

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
            "dataname": tune.grid_search(["Cora"]),#, "wisconsin","texas",'Cora', 'PubMed']),  # for ray tune, need abs path "CiteSeer","wikics","CS",'PubMed',
            "learning_rate": tune.grid_search([1e-4]),
            "weight_decay": tune.grid_search([1e-3]),
            "num_epochs": tune.grid_search([500]),
            "if_gat": tune.grid_search([True,False]),
            "if_linear": tune.grid_search([False]),
            "head": tune.grid_search([4]),
            "nhid": tune.grid_search([64]),
            "dropout_prob": tune.grid_search([0.1]),
            "num_scatter":tune.grid_search([3]),
            "diffusion": tune.grid_search(["randomwalk"]),

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
    parser.add_argument('--if_linear', type=str, default=False)
    parser.add_argument('--scattertype', type=str, default='sym')
    args = parser.parse_args()
    if args.action=="arg_train":
        arg_train(args)
    if args.action=="ray_train":
        ray_train()






