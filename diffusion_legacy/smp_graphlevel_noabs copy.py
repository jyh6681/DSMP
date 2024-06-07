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
from dataset_src.Graph_Dataset import *
import math
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, GELU, ELU, Dropout
from torch_geometric.nn import global_add_pool
import matplotlib.pyplot as plt
# %% function for early_stopping
import scipy.sparse as sp
import time
from sklearn.metrics import roc_auc_score
from numba import jit
# from ogb.graphproppred import PygGraphPropPredDataset
def adj_torch_sparse(A):
    ed = sp.coo_matrix(A)
    indices = np.vstack((ed.row, ed.col))
    index = torch.LongTensor(indices)
    values = torch.Tensor(ed.data)
    torch_sparse_mat = torch.sparse_coo_tensor(index, values, ed.shape)
    return torch_sparse_mat

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
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
            #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
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
    def __init__(self, num_features, nhid, num_classes, dropout_prob, hid_fc_dim, pRatio, head=4,layer=2,num_scatter=3,if_gat=True,if_linear=True):
        super(Net, self).__init__()
        self.dropout_prob = dropout_prob
        self.nhid = nhid
        self.pRatio = pRatio
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
        for i in range(num_scatter):#######each scatt layer is two step of scattering say it is MS and here we use num_scatter MS form the Net
            self.scatter_list.append(Scatter(in_channels=nhid, out_channels=nhid,dropout_prob=dropout_prob, if_gat=if_gat ,head=head,layer=layer))
        hidden_dim = [nhid] + hid_fc_dim + [num_classes]
        self.global_add_pool = global_add_pool
        fcList = [score_block(i, j, self.dropout_prob) for i, j in zip(hidden_dim[:-2], hidden_dim[1:])]
        fcList.append(nn.Sequential(nn.Linear(*hidden_dim[-2:]),nn.BatchNorm1d(hidden_dim[-1])))
        self.fc = nn.Sequential(*fcList)
    def forward(self, data):
        x_batch, edge_index_batch, batch,batch_scatter_list = (data.x).float(), data.edge_index, data.batch,data.scatter_list
        _, graph_size = torch.unique(batch, return_counts=True)
        scatter_list = self.scatter_deal(batch_scatter_list)
        self.scatter_edge_index_list, self.scatter_edge_attr_list = [], []
        for i in range(len(scatter_list)):##different layers correspond to different scatter matrix
            self.scatter_edge_index_list.append(scatter_list[i].coalesce().indices())
            self.scatter_edge_attr_list.append(scatter_list[i].coalesce().values())
        if self.if_linear == True:
            x_batch = self.lin_in(x_batch)
        hidden_rep = x_batch
        for index in range(self.num_scatter):
            # print("shape::::::",self.num_scatter,len(self.scatter_edge_index_list),self.scatter_edge_index_list[index].shape,edge_index_batch.shape) #6 4 torch.Size([2, 2972]) torch.Size([2, 2026])
            hidden_rep += self.scatter_list[index](hidden_rep,self.scatter_edge_index_list,self.scatter_edge_attr_list,edge_index_batch)
        h_pooled = self.global_add_pool(hidden_rep, batch)#grasspool(hidden_rep, graph_size, self.pRatio)
        x = self.fc(h_pooled)
        if self.num_classes == 1:
            return x.view(-1)
        else:
            return x
    @jit
    def adjConcat(self,a, b):
        '''
        concat two adj matrix along the diag
        '''
        lena = len(a)
        lenb = len(b)
        left = np.row_stack((a, np.zeros((lenb, lena))))  # get the left
        right = np.row_stack((np.zeros((lena, lenb)), b))  # get the right
        result = np.hstack((left, right))
        return result
    @jit
    def scatter_deal(self,batch_scatter_list):
        scatter_list = []
        ###select differrent batch in the same scatter scale mat (s[0][0],s[1][0],s[2][0]),s[0][1],s[1][1],s[2][1]
        start = time.time()
        for batch_graph in range(0, len(batch_scatter_list[0])):
            scatter_mat = batch_scatter_list[0][batch_graph]
            start1 = time.time()
            for index in range(1,len(batch_scatter_list)):
                mat = batch_scatter_list[index][batch_graph]
                scatter_mat = self.adjConcat(scatter_mat,mat)
            scatter_list.append(adj_torch_sparse(scatter_mat).to('cuda'))    ####scatter list contains low high1 high2 scatter mat
            print("adj:",time.time()-start1,len(batch_scatter_list), len(batch_scatter_list[0]))
        # print("batch time:",time.time()-start)
        return scatter_list
def test(model, loader, device,dataname):
    model.eval()
    correct = 0.
    loss = 0.
    auc=0.0
    batch=0
    for data in loader:
        data = data.to(device)
        out = model(data)
        out=F.softmax(out,dim=1)
        pred = out.max(dim=1)[1]#torch.argmax(out,dim=1)#
        correct += pred.eq(data.y).sum().item()
        loss += F.cross_entropy(out, data.y, reduction='sum').item()
        ###GET ROC_AUC_SCORE
        score = out.detach().cpu().numpy()
        true = F.one_hot(data.y).cpu().numpy()
        try:
            batch+=1
            auc+=roc_auc_score(y_true=true,y_score=score)
        except: ###except the case for all one class in y_true
            batch-=1
            pass

    return correct / len(loader.dataset), loss / len(loader.dataset),auc/batch

def objective(learning_rate=0.01, weight_decay=0.01, nhid=64,fc_dim=[64, 16],num_scatter=3,epochs=100, dropout_prob=0.5,batch_size=1024,num_reps=1,\
              pRatio=0.5,patience=20, if_gat=True, if_linear=True,head=4, dataname='Cora',appdix="randomwalk",diffusion="randomwalk"):
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    load_start = time.time()
    path = osp.join("/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/", 'data', "diffusion_data4.9", dataname)
    if dataname == 'qm7':
        dataset = SVDQM7(path)
        num_features = 5
        num_classes = 1
        loss_criteria = F.mse_loss
        dataset, mean, std = MyDataset(dataset, num_features)
    else:
        if dataname=="ogbg-molhiv":
            dataset = SVDGraphPropPredDataset(root=path,name=dataname)
            loss_criteria = F.cross_entropy
        elif dataname == 'COLLAB':  ##
            dataset = SVDTUD(path, name=dataname, pre_transform=T.OneHotDegree(max_degree=500))###add zero feature to 1000
            loss_criteria = F.cross_entropy
        elif dataname=="QM9":   ###19 targets regression
            dataset = SVDTUD(path, name=dataname,use_node_attr=True)
            loss_criteria = F.l1_loss
            mean=torch.min(dataset.data.y,dim=0).values.to(device)  #####qm9 use min max norm/mean=min
            k_index=[1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18]
            # for k in k_index:
            #     dataset.data.y[:,k]=dataset.data.y[:,k]/1000
            all_max = torch.max(dataset.data.y,dim=0).values.to(device)
            print("max:",all_max,mean)
            std=all_max-mean
            print("QM9:", dataset.data,mean.shape,std.shape)#,"\n","mean:",mean,"std:",std)
        else:
            dataset = SVDTUD(path, name=dataname) ###########################proteins remove first col
            loss_criteria = F.cross_entropy
        num_features = dataset.num_features
        print("features:", num_features)
        num_classes = dataset.num_classes
        print("num_calsses:", num_classes)
    print("load data cost:",time.time()-load_start)

    ###split train,val,test
    num_train, num_val = int(len(dataset) * 0.8), int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_train + num_val)

    # create results matrix
    epoch_train_loss = np.zeros((num_reps, epochs))
    epoch_train_acc = np.zeros((num_reps, epochs))
    epoch_valid_loss = np.zeros((num_reps, epochs))
    epoch_valid_acc = np.zeros((num_reps, epochs))
    epoch_test_loss = np.zeros((num_reps, epochs))
    epoch_test_acc = np.zeros((num_reps, epochs))
    saved_model_loss = np.zeros(num_reps)
    saved_model_acc = np.zeros(num_reps)

    save_dir = "/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/smp_noabs_results/"
    os.makedirs(save_dir,exist_ok=True)
    # training
    for r in range(num_reps):
        training_set, validation_set, test_set = random_split(dataset, [num_train, num_val, num_test])
        if dataname=="ogbg-molhiv":
            split_idx = dataset.get_idx_split()
            training_set, validation_set, test_set = dataset[split_idx["train"]],dataset[split_idx["valid"]],dataset[split_idx["test"]]
        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        model = Net(num_features, nhid, num_classes, dropout_prob, fc_dim, pRatio, num_scatter=num_scatter,head=head,if_gat=if_gat,
                       if_linear=if_linear).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=3,verbose=True)
        early_stopping = EarlyStopping(patience=patience, verbose=True,path=save_dir+ str(dataname) + \
                                            "if_gat" + str(if_gat) +"drop"+str(dropout_prob)+ "lr"+str(learning_rate)+"wd"+str(weight_decay)+"heads"+str(head)+str(args.appendix) + '.pth')
        # start training
        min_loss = 1e10
        patience = 0
        print("****** Start Rep {}: Training start ******".format(r + 1))
        for epoch in range(epochs):
            model.train()
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(device)
                start= time.time()
                out = model(data)
                # print("time for one pass:",time.time()-start)##0.1s,large batch cost about 10s
                if dataname=="QM9" or dataname=="qm7":
                    loss = loss_criteria(out, ((data.y-mean)/std), reduction='mean')
                else:
                    loss = loss_criteria(out, data.y.long(), reduction='mean')
                loss.backward()
                optimizer.step()
            if dataname == 'qm7' or dataname=="QM9":
                train_loss = qm_test_train(model, train_loader, device,mean,std,dataname)
                if dataname=="QM9":
                    val_loss,val_all = qm_test(model, val_loader, device, mean, std,dataname)
                    test_loss,test_all= qm_test(model, test_loader, device, mean, std,dataname)
                else:
                    val_loss = qm_test(model, val_loader, device, mean, std, dataname)
                    test_loss = qm_test(model, test_loader, device, mean, std, dataname)
                print("Epoch {}: Training loss: {:5f}, Validation loss: {:5f}, Test loss: {:.5f}".format(epoch + 1,train_loss,val_loss,test_loss))
            else:
                train_acc, train_loss,auc = test(model, train_loader, device,dataname)
                val_acc, val_loss,auc = test(model, val_loader, device,dataname)
                test_acc, test_loss,auc = test(model, test_loader, device,dataname)
                print("auc:",auc)
                epoch_train_acc[r, epoch], epoch_valid_acc[r, epoch], epoch_test_acc[
                    r, epoch] = train_acc, val_acc, test_acc
                print("Epoch {}: Training accuracy: {:.5f}; Validation accuracy: {:.5f}; Test accuracy: {:.5f}".format(epoch + 1, train_acc, val_acc, test_acc))
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
        # model = Net(num_features, nhid, num_classes, dropout_prob, fc_dim, pRatio, head=head,if_gat=if_gat,if_linear=if_linear).to(device)
        # model.load_state_dict(torch.load("/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/graph_results/diffusion_" + str(dataname) + \
        #                                             "if_gat" + str(if_gat) +"drop"+str(dropout_prob)+ "lr"+str(learning_rate)+"wd"+str(weight_decay)+"heads"+str(head)+str(args.appendix) + '.pth'))
        if dataname == 'qm7'or dataname=="QM9":
            if dataname == "QM9":
                test_loss, test_all = qm_test(model, test_loader, device, mean, std, dataname)
                test_acc = 1-test_loss
            else:
                test_loss = qm_test(model, test_loader, device, mean, std, dataname)
                test_acc = 1 - test_loss
            print("args: " + str(dataname) + "if_gat" + str(if_gat) + "drop" + str(dropout_prob) + "lr" + str(learning_rate) + "wd" + str(weight_decay))
            print("Test Loss: {:.5f}".format(test_loss))
        else:
            test_acc, test_loss,auc = test(model, test_loader, device,dataname)
            saved_model_acc[r] = test_acc
            print("args: ",args)
            print("Test accuracy: {:.5f}".format(test_acc))
            if dataname=="ogbg-molhiv":
                print("test auc:",auc)
        saved_model_loss[r] = test_loss

    # save the results
    plt_path = "/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/graph_results/diffusion_test_" + str(dataname)+"if_gat" + str(if_gat) \
                                         +"num_scatter"+str(num_scatter)+"drop"+str(dropout_prob)+str(round(test_acc, 4)) + "lr" + str(learning_rate) + "wd" + str(
            weight_decay)+str(args.appendix)
    epoch_list = range(0, epochs)
    ax = plt.gca()
    ax.grid()
    if dataname == "qm7" or dataname=="QM9":
        ax.set_ylim(0, 1.2 * max(epoch_test_loss[0, :]))
        plt.plot(epoch_list, epoch_train_loss.squeeze(), 'r-', label="train-loss")
        plt.plot(epoch_list, epoch_valid_loss.squeeze(), 'b-', label="val-loss")
        plt.plot(epoch_list, epoch_test_loss.squeeze(), 'g-', label="test-loss/best=" + str(round(test_loss, 2)))
        plt.xlabel("epoches")
        # plt.ylabel('Training psnr&Val psnr')
        plt.legend()
        plt.savefig(plt_path+ 'curve.png', bbox_inches='tight')
        if dataname=="QM9":
            print("test result:",test_all)
            return round(test_loss, 2),test_all
        else:
            return round(test_loss, 2)
    else:
        ax.set_ylim(0, 1)
        plt.plot(epoch_list, epoch_train_acc.squeeze(), 'r-', label="train-acc")
        plt.plot(epoch_list, epoch_valid_acc.squeeze(), 'b-', label="val-acc")
        plt.plot(epoch_list, epoch_test_acc.squeeze(), 'g-', label="test-acc/best=" + str(round(test_acc, 4)))
        plt.xlabel("epoches")
        # plt.ylabel('Training psnr&Val psnr')
        plt.legend()
        plt.savefig( plt_path+ 'curve.png', bbox_inches='tight')
        if dataname=="ogbg-molhiv":
            return auc
        else:
            return round(test_acc, 4)
    # plt.show()
    # np.savez(str(appdix) + '.npz',
    #          epoch_train_loss=epoch_train_loss,
    #          epoch_train_acc=epoch_train_acc,
    #          epoch_valid_loss=epoch_valid_loss,
    #          epoch_valid_acc=epoch_valid_acc,
    #          epoch_test_loss=epoch_test_loss,
    #          epoch_test_acc=epoch_test_acc,
    #          saved_model_loss=saved_model_loss,
    #          saved_model_acc=saved_model_acc)

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
    ray.init()

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
            "dataname": tune.grid_search(["PROTEINS","COLLAB","Mutagenicity","NCI1","DD"]), #,",,"qm7","ogbg-molhiv","NCI1", DD,ogbg-molhiv,"NCI1","ZINC_full","IMDB-MULTI"]),#DD, Mutagenicity,"NCI1",'Cora', 'PubMed']),  # for ray tune, need abs path ogbg-molhiv
            "learning_rate": tune.grid_search([1e-4]),
            "weight_decay": tune.grid_search([1e-4]),
            "if_gat": tune.grid_search([True,False]),
            "if_linear": tune.grid_search([False]),
            "batch_size":tune.grid_search([32]),
            "epochs": tune.grid_search([100]),
            "head": tune.grid_search([4]),####no gat no head
            "nhid": tune.grid_search([16]),
            "dropout_prob": tune.grid_search([0.1]),
            "num_scatter": tune.grid_search([3]),
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
    return df
def arg_train(args):
    df = objective(learning_rate=args.lr, weight_decay=args.wd, nhid=args.nhid,epochs=args.epochs, dropout_prob=args.drop, batch_size=args.batch_size, head=4,if_linear=args.if_linear,if_gat=args.if_gat,
                     dataname=args.dataname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default=ray_train, help="ray or arg train")
    parser.add_argument('--dataname', type=str, default='qm7',
                        help='name of dataset (default: PROTEINS)')
    parser.add_argument('--reps', type=int, default=3,
                        help='number of repetitions (default: 10)')
    parser.add_argument("-b",'--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument("-e",'--epochs', type=int, default=100,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--patience', type=int, default=20,
                        help='early stopping criteria (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay (default: 5e-3)')
    parser.add_argument('--conv_hid', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--num_conv_layer', type=int, default=2,
                        help='number of hidden mlp layers in a conv layer (default: 2)')
    parser.add_argument('--drop', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
    parser.add_argument("--fc_dim", type=int, nargs="*", default=[64, 16],
                        help="dimension of fc hidden layers (default: [64])")
    parser.add_argument('--pRatio', type=float, default=0.8,
                        help='the ratio of info preserved in the Grassmann subspace (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--filename', type=str, default='results',
                        help='filename to store results and the model (default: results)')
    parser.add_argument('--if_gat', type=str, default=True)
    parser.add_argument('--if_linear', type=str, default=False)
    parser.add_argument('-a','--appendix', type=str, default="7.26")

    args = parser.parse_args()

    if args.action=="arg_train":
        arg_train(args)
    if args.action=="ray_train":
        ray_train()



