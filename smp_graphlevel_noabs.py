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
import statistics
import time
from sklearn.metrics import roc_auc_score
from numba import jit
from model.DSMP import *
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
# from ogb.graphproppred import PygGraphPropPredDataset
num_gpus = torch.cuda.device_count()
print(f"可见的GPU数量：{num_gpus}")
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

def test(model, loader, device,dataname):
    model.eval()
    correct = 0.
    loss = 0.
    auc=0.0
    auc_roc = BinaryAUROC().to(device)
    batch=0
    for data in loader:
        data = data.to(device)
        out = model(data)
        out=F.softmax(out,dim=1)
        pred = out.max(dim=1)[1]#torch.argmax(out,dim=1)#
        correct += pred.eq(data.y).sum().item()
        loss += F.cross_entropy(out, data.y, reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset),0#auc/batch

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
        elif dataname == 'COLLAB' or dataname =="IMDB-BINARY":  ##
            dataset = SVDTUD(path, name=dataname, pre_transform=T.OneHotDegree(max_degree=492))###add zero feature to 1000
            loss_criteria = F.cross_entropy
        elif dataname =="IMDB-BINARY":  ##
            dataset = SVDTUD(path, name=dataname, pre_transform=T.OneHotDegree(max_degree=150))###add zero feature to 1000
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
    # print("load data cost:",time.time()-load_start)

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
    reps_acc = []
    # training
    for r in range(num_reps):
        training_set, validation_set, test_set = random_split(dataset, [num_train, num_val, num_test])
        # if dataname=="ogbg-molhiv":
        #     split_idx = dataset.get_idx_split()
        #     training_set, validation_set, test_set = dataset[split_idx["train"]],dataset[split_idx["valid"]],dataset[split_idx["test"]]
        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        model = Net(num_features, nhid, num_classes, dropout_prob, fc_dim, pRatio, num_scatter=num_scatter,head=head,if_gat=if_gat,
                       if_linear=if_linear).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=3,verbose=False)
        early_stopping = EarlyStopping(patience=patience, verbose=False,path=save_dir+ str(dataname) + \
                                            "if_gat" + str(if_gat) +"drop"+str(dropout_prob)+ "lr"+str(learning_rate)+"wd"+str(weight_decay)+"heads"+str(head)+str(args.appendix) + '.pth')
        # start training
        best_acc = 0
        our_patience = 0
        best_dict = model.state_dict()
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
                if best_acc<val_acc:
                    best_dict = model.state_dict()
                    best_acc=val_acc
                epoch_train_acc[r, epoch], epoch_valid_acc[r, epoch], epoch_test_acc[
                    r, epoch] = train_acc, val_acc, test_acc
            # if epoch%10==0:
            #     print(dataname+" on Epoch {}: Training accuracy: {:.4f}; Validation accuracy: {:.4f}; Test accuracy: {:.4f};Test auc: {:.4f}".format(epoch + 1, train_acc, val_acc, test_acc,auc))
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
        model.load_state_dict(best_dict)
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
            # print("args: ",args)
            # print("Test accuracy: {:.4f}".format(test_acc))
            reps_acc.append(test_acc)
            # if dataname=="ogbg-molhiv":
            #     print("test auc:",auc)
        saved_model_loss[r] = test_loss
    # print(dataname, "mean and var:", round(statistics.mean(reps_acc),4),round(statistics.variance(reps_acc),4))
    print("test:",test_acc,args)
    if dataname=="ogbg-molhiv":
        return auc
    else:
        return test_acc#round(statistics.mean(reps_acc),4),round(statistics.variance(reps_acc),4)

def ray_train(args):
    class ExperimentTerminationReporter(CLIReporter):
        def should_report(self, trials, done=True):
            """Reports only on experiment termination."""
            return done

    def training_function(config):
        acc = objective(**config)
        tune.report(acc=acc)

    # ray.shutdown()
    # ray.init(ignore_reinit_error=True)
    ray.init(address='auto')

    sched = ASHAScheduler(metric="mean_accuracy", mode="max")
    analysis = tune.run(training_function,
        scheduler=sched,
        config={
            "dataname": tune.grid_search(["COLLAB"]), #,,",,"qm7","ogbg-molhiv","NCI1", DD,ogbg-molhiv,"NCI1","ZINC_full","IMDB-MULTI"]),#DD, Mutagenicity,"NCI1",'Cora', 'PubMed']),  # for ray tune, need abs path ogbg-molhiv
            ###TUdataset "ENZYMES","MUTAG","IMDB",NCI1", "DD"
            "learning_rate": tune.grid_search([1e-2,1e-3,1e-4]),
            "weight_decay": tune.grid_search([1e-3,1e-4]),
            "if_gat": tune.grid_search([True]),
            "if_linear": tune.grid_search([False]),
            "batch_size":tune.grid_search([8]),#args.batch_size]),
            "epochs": tune.grid_search([200]),
            "head": tune.grid_search([4]),####no gat no head
            "nhid": tune.grid_search([16,128]),
            "dropout_prob": tune.grid_search([0.0,0.5]),
            "num_scatter": tune.grid_search([3]),
        },
        progress_reporter=ExperimentTerminationReporter(),
        resources_per_trial={'gpu': 1, 'cpu': 4})
    analysis.results_df.to_csv("result.csv")
    print("Best config: ", analysis.get_best_config(metric="acc", mode="max"))
    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    ray.shutdown()
    return df

def arg_train(args):
    df = objective(learning_rate=args.lr, weight_decay=args.wd, nhid=args.nhid,epochs=args.epochs, dropout_prob=args.drop, batch_size=args.batch_size, head=4,if_linear=args.if_linear,if_gat=args.if_gat,
                     dataname=args.dataname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, default=ray_train, help="ray or arg train")
    parser.add_argument('--dataname', type=str, default='IMDB-BINARY',
                        help='name of dataset (default: PROTEINS)')
    parser.add_argument('--single_data', action='store_true')
    parser.add_argument('--reps', type=int, default=3,
                        help='number of repetitions (default: 10)')
    parser.add_argument("-b",'--batch_size', type=int, default=8,
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
        ray_train(args)



