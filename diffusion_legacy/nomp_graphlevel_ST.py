import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import argparse
import os.path as osp
from Graph_Dataset import *
import math
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, GELU, ELU, Dropout
from mSVD import mSVD
from torch_geometric.nn import  global_add_pool
import matplotlib.pyplot as plt
# %% function for early_stopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), "./results/"+self.path)
        self.val_loss_min = val_loss
# %% FC Layer
def score_block(input_dim, output_dim, dropout):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(),
                         nn.Dropout(dropout))
# %% GrassPool only for gin
def grasspool(hid, graph_sizes, pRatio):
    """
    cur_node_embeddings: hidden rep for a single graph g_i
    hid: hidden rep of batch_graph to be transformed
    graph_sizes: a list of individual graph node size
    """
    graph_sizes = graph_sizes.tolist()
    node_embeddings = torch.split(hid, graph_sizes)
    ### create an autograd-able variable
    batch_graphs = torch.zeros(len(graph_sizes), int(hid.shape[1] * (hid.shape[1] + 1) / 2)).to(hid.device)

    for g_i in range(len(graph_sizes)):
        cur_node_embeddings = node_embeddings[g_i]
        U, S, V = mSVD.apply(cur_node_embeddings.t())
        k = sum(S > pRatio).item()
        subspace_sym = torch.matmul(U[:, :k], U[:, :k].t())
        ### flatten
        idx = torch.triu_indices(subspace_sym.shape[0], subspace_sym.shape[0])
        cur_graph_tri_u = subspace_sym[idx[0], idx[1]]
        batch_graphs[g_i] = cur_graph_tri_u.flatten()
    return batch_graphs
# %%
# =============================================================================
# scattering transform
''' get all scattering coefficients '''
##################################construction of Scattering transform####################
class ScatterSplit(MessagePassing):
    def __init__(self, in_channels, out_channels,dropout_prob=0.5, if_mlp=True, head=2,layer=3,J=3,learn=True):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

        self.if_mlp = if_mlp
        self.dropout_prob = dropout_prob

        self.act_drop = Seq(
            ELU(),
            Dropout(dropout_prob),
        )
        self.J=J
        self.in_channels = in_channels
        self.learn=learn
        self.layer = layer
        self.num_gat = 2**layer - 1  # layer*(J-1) ####multiple times1+(3-1)+(3-1)*3=9,low pass1,2,4,8,16high pass2,4,8,16,32
        if self.learn == True:
            self.conv_list = nn.ModuleList([GATConv(self.in_channels, out_channels, heads=head, dropout=dropout_prob) for i in range(0, self.num_gat)])
            self.mlp_list = nn.ModuleList([Seq(ELU(), Dropout(dropout_prob), Linear(head * out_channels, out_channels),nn.BatchNorm1d(out_channels)) for i in range(0, self.num_gat)])
        if self.learn == False:
            self.mlp_learn = nn.ModuleList([Seq(nn.BatchNorm1d(self.in_channels), ELU(), Dropout(dropout_prob), Linear(self.in_channels, out_channels)) for i in range(0, self.num_gat)])
        self.highpass = False

    def forward(self, x_in, edge_index):  ###x_in is the list of scattering coef (numpy data)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [2, E], i.e. the ajacent matrix at one single level
        #x = getRep(x_in, self.lamb, self.V, layer=self.layer)
        split_size = int(x_in.shape[1]/self.num_gat)
        rep_list = torch.split(x_in,split_size ,dim=1)
        #print("x_list:",x_in.shape,len(rep_list),rep_list[0].shape,self.num_gat)
        index=0
        x_out= []
        for rep in rep_list:
            if self.learn==True:
                x_out.append(self.mlp_list[index](self.conv_list[index](rep, edge_index)))
            else:
                x_out.append(self.mlp_learn[index](rep))
            index+=1
        x_out=torch.cat(x_out,dim=1)
        #print("list:::",len(rep_list),x_out.shape,x_in.shape)
        return x_out

class Scatter(MessagePassing):
    def __init__(self, in_channels, out_channels,dropout_prob=0.5, if_mlp=True, head=2,layer=3,J=3,learn=True):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

        self.if_mlp = if_mlp
        self.dropout_prob = dropout_prob

        self.act_drop = Seq(
            ELU(),
            Dropout(dropout_prob),
        )
        self.J=J
        self.in_channels = (2 ** layer - 1)*in_channels
        self.learn=learn
        self.layer = layer
        self.num_gat = 1#2 ** layer - 1  # layer*(J-1) ####multiple times1+(3-1)+(3-1)*3=9,low pass1,2,4,8,16high pass2,4,8,16,32
        if self.learn == True:
            self.conv_list = nn.ModuleList([GATConv(self.in_channels, out_channels, heads=head, dropout=dropout_prob) for i in range(0, self.num_gat)])
            self.mlp_list = nn.ModuleList([Seq(ELU(), Dropout(dropout_prob), Linear(head * out_channels, out_channels),nn.BatchNorm1d(out_channels)) for i in range(0, self.num_gat)])
        if self.learn == False:
            self.mlp_learn = nn.ModuleList([Seq(nn.BatchNorm1d(self.in_channels), ELU(), Dropout(dropout_prob), Linear(self.in_channels, out_channels)) for i in range(0, self.num_gat)])
        self.highpass = False

    def forward(self, x_in, edge_index):  ###x_in is the list of scattering coef (numpy data)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [2, E], i.e. the ajacent matrix at one single level
        index=0
        if self.learn==True:
            x_out=self.mlp_list[index](self.conv_list[index](x_in, edge_index))
        else:
            x_out=self.mlp_learn[index](x_in)
        return x_out
class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, dropout_prob, hid_fc_dim, pRatio, head=2,layer=3,learn=False,split=0,dataset="qm7"):
        super(Net, self).__init__()
        self.dropout_prob = dropout_prob
        self.nhid = nhid
        self.pRatio = pRatio
        self.split = split
        self.dataset = dataset
        if split==1:
            self.scatter = ScatterSplit(in_channels=num_features, out_channels=nhid,dropout_prob=dropout_prob, head=head,layer=layer,learn=learn)
            hidden_dim = [(2**layer-1)*nhid] + hid_fc_dim + [num_classes]
        if split == 0:
            self.scatter = Scatter(in_channels=num_features, out_channels=nhid, dropout_prob=dropout_prob,head=head, layer=layer, learn=learn)
            hidden_dim = [nhid] + hid_fc_dim + [num_classes]
        self.global_add_pool = global_add_pool
        fcList = [score_block(i, j, self.dropout_prob) for i, j in zip(hidden_dim[:-2], hidden_dim[1:])]
        fcList.append(nn.Sequential(nn.Linear(*hidden_dim[-2:]),nn.BatchNorm1d(hidden_dim[-1])))
        self.fc = nn.Sequential(*fcList)
    def forward(self, data):
        x_batch, edge_index_batch, batch = data.x, data.edge_index, data.batch
        # if self.dataset=="qm7":
        #     x_batch = torch.from_numpy(np.concatenate(x_batch,axis=0)).to("cuda")
        _, graph_size = torch.unique(batch, return_counts=True)
            #print("batch:",data,x_batch.shape,edge_index_batch.shape,batch.shape,graph_size)
        hidden_rep = self.scatter(x_batch,edge_index_batch)
        h_pooled = self.global_add_pool(hidden_rep, batch)#grasspool(hidden_rep, graph_size, self.pRatio)
        x = self.fc(h_pooled)
        if num_classes == 1:
            return x.view(-1)
        else:
            return x

def test(model, loader, device):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.cross_entropy(out, data.y, reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset)

def pre_scatter_transform(data):
    edge_index = data.edge_index
    num_nodes = data.x.shape[0]
    L = get_laplacian(edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())),
                          shape=(num_nodes, num_nodes))
    lamb, V = np.linalg.eigh(L.toarray())
    x_in = data.x
    scatter_rep = getRep(x_in, lamb, V, layer=3)  ##(2**layer-1)*num_nodes
    #print("type::::", type(data.x), data.x.shape, scatter_rep.shape)
    data = Data(x= scatter_rep,edge_index=edge_index,y=data.y)
    #data.x = torch.from_numpy(scatter_rep)
    return data
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qm7',
                        help='name of dataset (default: PROTEINS)')
    parser.add_argument('--reps', type=int, default=1,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument("-e",'--epochs', type=int, default=100,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--patience', type=int, default=20,
                        help='early stopping criteria (default: 20)')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=1e-3,
                        help='weight decay (default: 5e-3)')
    parser.add_argument('--conv_hid', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--num_conv_layer', type=int, default=2,
                        help='number of hidden mlp layers in a conv layer (default: 2)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
    parser.add_argument("--fc_dim", type=int, nargs="*", default=[64, 16],
                        help="dimension of fc hidden layers (default: [64])")
    parser.add_argument('--pRatio', type=float, default=0.8,
                        help='the ratio of info preserved in the Grassmann subspace (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--filename', type=str, default='results',
                        help='filename to store results and the model (default: results)')
    parser.add_argument('--learn', type=bool, default=True)
    parser.add_argument('--split', type=int, default=0)

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    path = osp.join(osp.abspath(''), 'data', args.dataset)
    import scipy
    if args.dataset == 'qm7':
        data = scipy.io.loadmat("./data/qm7.mat")
        #print("show data:",data["X"].shape,data["R"].shape,data["P"].shape,data["Z"].shape,data["T"].shape)
        dataset = SVDQM7(path)
        num_features = 5
        num_classes = 1
        loss_criteria = F.mse_loss
        dataset, mean, std = MyDataset(dataset, num_features)
    else:
        if args.dataset == 'COLLAB':   ###protein
            dataset = SVDTUD(path, transform=T.OneHotDegree(max_degree=1000))
            num_features = int(dataset.num_features/7)
        else:
            dataset = SVDTUD(path, name=args.dataset)
            num_features = 4
        num_classes = dataset.num_classes
        loss_criteria = F.cross_entropy

    num_train, num_val = int(len(dataset) * 0.8), int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_train + num_val)

    # Parameter Setting
    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.conv_hid
    epochs = args.epochs
    num_reps = args.reps

    # create results matrix
    epoch_train_loss = np.zeros((num_reps, epochs))
    epoch_train_acc = np.zeros((num_reps, epochs))
    epoch_valid_loss = np.zeros((num_reps, epochs))
    epoch_valid_acc = np.zeros((num_reps, epochs))
    epoch_test_loss = np.zeros((num_reps, epochs))
    epoch_test_acc = np.zeros((num_reps, epochs))
    saved_model_loss = np.zeros(num_reps)
    saved_model_acc = np.zeros(num_reps)

    # training
    for r in range(num_reps):
        training_set, validation_set, test_set = random_split(dataset, [num_train, num_val, num_test])

        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        model = Net(num_features, nhid, num_classes, args.drop_ratio, args.fc_dim, args.pRatio,learn=args.learn,
                    split=args.split,dataset=args.dataset.lower()).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5,
                                                               verbose=True)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.filename + '_latest.pth')

        # start training
        min_loss = 1e10
        patience = 0
        print("****** Rep {}: Training start ******".format(r + 1))
        for epoch in range(epochs):
            model.train()
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(device)
                out = model(data)
                loss = loss_criteria(out, data.y, reduction='sum')
                loss.backward()
                optimizer.step()
            if args.dataset == 'qm7':
                train_loss = qm7_test_train(model, train_loader, device)
                val_loss = qm7_test(model, val_loader, device, mean, std)
                test_loss = qm7_test(model, test_loader, device, mean, std)
                print("Epoch {}: Training loss: {:5f}, Validation loss: {:5f}, Test loss: {:.5f}".format(epoch + 1,train_loss,val_loss,test_loss))
            else:
                train_acc, train_loss = test(model, train_loader, device)
                val_acc, val_loss = test(model, val_loader, device)
                test_acc, test_loss = test(model, test_loader, device)
                epoch_train_acc[r, epoch], epoch_valid_acc[r, epoch], epoch_test_acc[r, epoch] = train_acc, val_acc, test_acc
                print("Epoch {}: Training accuracy: {:.5f}; Validation accuracy: {:.5f}; Test accuracy: {:.5f}".format(
                    epoch + 1, train_acc, val_acc, test_acc))
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping \n")
                break
            scheduler.step(val_loss)

            epoch_train_loss[r, epoch] = train_loss
            epoch_valid_loss[r, epoch] = val_loss
            epoch_test_loss[r, epoch] = test_loss

        # Test
        print("****** Test start ******")
        model = Net(num_features, nhid, num_classes, args.drop_ratio, args.fc_dim, args.pRatio,learn=args.learn,
                    split=args.split,dataset=args.dataset.lower()).to(device)
        model.load_state_dict(torch.load("./results/"+args.filename + '_latest.pth'))
        if args.dataset == 'qm7':
            test_loss = qm7_test(model, test_loader, device, mean, std)
            print("Test Loss: {:.5f}".format(test_loss))
        else:
            test_acc, test_loss = test(model, test_loader, device)
            saved_model_acc[r] = test_acc
            print("Test accuracy: {:.5f}".format(test_acc))
        saved_model_loss[r] = test_loss

    # save the results
    epoch_list  = range(0,epochs)
    ax = plt.gca()
    ax.grid()
    ax.set_ylim(0, 2*min(epoch_test_loss[0,:]))
    if args.dataset=="qm7":
        plt.plot(epoch_list, epoch_train_loss.squeeze(), 'r-', label="train-loss")
        plt.plot(epoch_list, epoch_valid_loss.squeeze(), 'b-', label="val-loss")
        plt.plot(epoch_list, epoch_test_loss.squeeze(), 'g-', label="test-loss/best=" + str(round(test_loss, 2)))
    else:
        plt.plot(epoch_list, epoch_train_acc.squeeze(), 'r-', label="train-acc")
        plt.plot(epoch_list, epoch_valid_acc.squeeze(), 'b-', label="val-acc")
        plt.plot(epoch_list, epoch_test_acc.squeeze(), 'g-', label="test-acc/best=" + str(round(test_acc, 2)))
    plt.xlabel("epoches")
    # plt.ylabel('Training psnr&Val psnr')
    plt.legend()
    plt.savefig("./results/test_" + str(round(test_loss, 2)) + "lr" + str(args.lr) + "wd" + str(args.wd) + 'curve.png', bbox_inches='tight')
    plt.show()
    np.savez(args.filename + '.npz',
             epoch_train_loss=epoch_train_loss,
             epoch_train_acc=epoch_train_acc,
             epoch_valid_loss=epoch_valid_loss,
             epoch_valid_acc=epoch_valid_acc,
             epoch_test_loss=epoch_test_loss,
             epoch_test_acc=epoch_test_acc,
             saved_model_loss=saved_model_loss,
             saved_model_acc=saved_model_acc)
