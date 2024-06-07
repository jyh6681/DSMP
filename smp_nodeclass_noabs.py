from scipy import sparse
from scipy.sparse.linalg import lobpcg
from torch_geometric.nn import GCNConv, GATConv,GINConv
from torch_geometric.datasets import Planetoid, WikiCS,Coauthor,AttributedGraphDataset,Reddit
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, degree
import os.path as osp
import random
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import scipy
from deeprobust.graph.utils import *
from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
import networkx as nx
import argparse
from numba import jit
from dataset_src.new_dataset import *
from model.DSMP_Node import Net
import os
parser = argparse.ArgumentParser()
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
os.environ["CUDA_VISIBLE_DEVICES"] = "7,6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def graph_reader(edge_list):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(edge_list)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

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

D1 = lambda x: np.cos(x / 2)
D2 = lambda x: np.sin(x / 2)
DFilters = [D1, D2]
RFilters = [D1, D2]
Lev = 2  # level of transform
s = 2  # dilation scale
n = 2  # n - 1 = Degree of Chebyshev Polynomial Approximation
r = len(DFilters)

def adj_torch_sparse(A):
    ed = sp.coo_matrix(A)
    indices = np.vstack((ed.row, ed.col))
    index =  torch.Tensor(indices)##torch.LongTensor(indices).to(device)
    values = torch.Tensor(ed.data)
    torch_sparse_mat = torch.sparse_coo_tensor(index, values, ed.shape).to(device)
    return torch_sparse_mat

# @jit
def scatter_index_value(scatter_mat):
    scatter_edge = []
    scatter_attr = []
    for index in range(0,len(scatter_mat)): ##low, high requency
        mat = adj_torch_sparse(scatter_mat[index])
        scatter_edge.append(mat.coalesce().indices())
        scatter_attr.append(mat.coalesce().values())
    return scatter_edge,scatter_attr


def chebyshev_mat(edge_index):  # to replace propLayer,V is the adjacent matrix
    num_nodes = edge_index.max().item() + 1
    L = get_laplacian(edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))
    lobpcg_init = np.random.rand(num_nodes, 1)
    lambda_max, _ = lobpcg(L, lobpcg_init, maxiter=50)
    lambda_max = lambda_max[0]
    J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # dilation level to start the decomposition
    d = get_operator(L, DFilters, n, s, J, Lev)
    # enhance sparseness of the matrix operators (optional)
    # d[np.abs(d) < 0.001] = 0.0
    # store the matrix operators (torch sparse format) into a list: row-by-row
    d_list = list()
    for l in range(Lev):
        for i in range(r):
            # if l==Lev-1:
            #     d[i, l][np.abs(d[i, l]) < 0.001] = 0.0
            d_list.append(d[i, l].toarray())
    return d_list #W_{0,J},W_{1,1},W_{1,J} total number is K*L-1


def objective(args, learning_rate= 1e-4, weight_decay=1e-4, nhid=16,fc_dim=[64, 16],num_scatter=3,epochs=100, dropout_prob=0.1,batch_size=1024,num_reps=1,\
              pRatio=0.5,patience=20, if_gat=True, if_linear=False,head=4, dataname='Cora',appdix="framelet",diffusion="randomwalk"):
    rootname = osp.join("/home/jyh_temp1/Downloads/GraphLocalDenoising/", dataname)
    if dataname.lower() == 'wikics':
        print("load wikics",WikiCS(root=rootname).num_classes)
        dataset = WikiCS(root=rootname)
        num_class = dataset.num_classes
        data = dataset[0]
        clean_data = data.clone().to(device)
        clean_data['train_mask'] = clean_data['train_mask'][:, 0]
        clean_data['val_mask'] = clean_data['val_mask'][:, 0]
    if dataname.lower() == "ogbn-arxiv":
        NoTrans_dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/home/jyh_temp1/Downloads/GraphLocalDenoising/arxiv/',)
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='/home/jyh_temp1/Downloads/GraphLocalDenoising/arxiv/', transform=T.ToSparseTensor())
        num_class = dataset.num_classes
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"]   , split_idx["test"]
        train_mask = torch.zeros((dataset[0].x.shape[0],), dtype=torch.bool)
        train_mask[train_idx] = 1
        val_mask = torch.zeros((dataset[0].x.shape[0],), dtype=torch.bool)
        val_mask[val_idx] = 1
        test_mask = torch.zeros((dataset[0].x.shape[0],), dtype=torch.bool)
        test_mask[test_idx] = 1
        print("mask:", num_class,dataset[0].x.shape,train_mask.shape, train_mask,train_idx.shape,val_idx.shape,test_idx.shape)
        data = Data(x=dataset[0].x, edge_index=NoTrans_dataset[0].edge_index, y=dataset[0].y.squeeze(1), train_mask=train_mask,val_mask=val_mask, test_mask=test_mask)
        # data = Data(x=dataset[0].x, edge_index=NoTrans_dataset[0].edge_index, y=dataset[0].y.squeeze(1), train_mask=train_idx,val_mask=val_idx, test_mask=test_idx)
        clean_data = data.clone().to(device)
    if dataname.lower() == 'cs':
        dataset = Coauthor(root="/home/jyh_temp1/Downloads/GraphLocalDenoising/CS", name=dataname)
        num_class = dataset.num_classes
        train_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        train_mask[0:300] = 1
        print("train:", train_mask.shape, dataset[0].y.shape)
        val_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        val_mask[300:500] = 1
        test_mask = torch.zeros_like(dataset[0].y, dtype=torch.bool)
        test_mask[500:1500] = 1
        data = Data(x=dataset[0].x, edge_index=dataset[0].edge_index, y=dataset[0].y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        clean_data = data.clone().to(device)
        if_linear = True
    if dataname.lower() == 'cora' or dataname.lower() == 'pubmed' or dataname.lower() == 'citeseer':
        print("load Planet data")
        dataset = Planetoid(root=rootname, name=dataname)
        num_class = dataset.num_classes
        data = dataset[0]
        clean_data = data.clone().to(device)
    if dataname.lower() == 'wisconsin' or dataname.lower() == 'texas':
        dataset = HetroDataSet(root=rootname, name=dataname)
        num_class = dataset.num_classes
        data = dataset[0]
        clean_data = data.clone().to(device)
        print("clean mask:", torch.sum(clean_data.train_mask.int()), torch.sum(clean_data.val_mask.int()),
              torch.sum(clean_data.test_mask.int()))
    #######
    num_features = data.x.shape[1]
    num_nodes = data.x.shape[0]
    edges = clean_data.edge_index
    # if dataname.lower() == "ogbn-arxiv":
    #     edges = dataset[0].adj_t.to_symmetric().to(device)
    #     data.edge_index = edges
    ##Framelet elements
    scatter_list = chebyshev_mat(data.edge_index) ####list
    scatter_index,scatter_value = scatter_index_value(scatter_list)

    # training

    if args.datatype.lower() == "noisy":
        noise_data = np.load(
            "/home/jyh_temp1/Downloads/GraphLocalDenoising/" + str(dataname.lower()) + "_injection/" + str(dataname.lower()) + "noise_mat_ratiio" + str(
                0.5) + ".npy")
        if args.mettackdata==1:
            noise_data = np.load("/home/jyh_temp1/Downloads/GraphLocalDenoising/Mettack_data/"+str(dataname.lower())+"/"+ str(
                    dataname.lower()) + "mettack_feat" + ".npy")
        noise_data = torch.from_numpy(noise_data).to(device)
        data.x = noise_data
        data = data.to(device)
    if args.datatype == "clean":
        data = clean_data.to(device)

    # create result matrices
    num_epochs = epochs
    num_reps = args.reps
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
    acc_list = []
    save_dir = "/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/Node_smp_noabs_results/"
    os.makedirs(save_dir,exist_ok=True)
    learning_rate=args.lr
    for rep in range(num_reps):
        # print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0
        model = Net(num_features, nhid, num_class, dropout_prob, fc_dim, pRatio, num_scatter=num_scatter,head=head,if_gat=if_gat,
                       if_linear=if_linear).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=3,verbose=False)
        early_stopping = EarlyStopping(patience=patience, verbose=False,path=save_dir+ str(dataname) + \
                                            "if_gat" + str(if_gat) +"drop"+str(dropout_prob)+ "lr"+str(learning_rate)+"wd"+str(weight_decay)+"heads"+str(head) + '.pth')
        # start training
        best_acc = 0
        our_patience = 0
        best_dict = model.state_dict()
        print("****** Start Rep {}: Training start ******".format(r + 1))
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data, scatter_index,scatter_value)
            loss = F.nll_loss(out[clean_data.train_mask], data.y[clean_data.train_mask])
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # evaluation mode
            with torch.no_grad():
                model.eval()
                out = model(data, scatter_index,scatter_value)
                for i, mask in clean_data('train_mask', 'val_mask', 'test_mask'):
                    pred =  out[mask].max(dim=1)[1]
                    correct = float(pred.eq(data.y[mask]).sum().item())
                    e_acc = correct / mask.sum().item()
                    epoch_acc[i][rep, epoch] = e_acc
                    e_loss = F.nll_loss(out[mask], data.y[mask])
                    epoch_loss[i][rep, epoch] = e_loss
            # print out results
            # print('Epoch: {:3d}'.format(epoch + 1),
            #       'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
            #       'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
            #       'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
            #       'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
            #       'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
            #       'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            # save model
            if epoch_acc['val_mask'][rep, epoch] > max_acc:
                torch.save(model.state_dict(), save_dir+ str(dataname) + \
                                            "if_gat" + str(if_gat) +"drop"+str(dropout_prob)+ "lr"+str(learning_rate)+"wd"+str(weight_decay)+"heads"+str(head) + '.pth')
                # print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                max_acc = epoch_acc['val_mask'][rep, epoch]
                record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        acc_list.append(record_test_acc)
        # print("time cost:",time.time()-start)
        # print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc,record_test_acc))
    print(
        '***************************************************************************************************************************')
    print('Average test accuracy over {0:2d} reps: {1:.4f} with stdev {2:.4f}'.format(num_reps,
                                                                                      np.mean(saved_model_test_acc),
                                                                                      np.std(saved_model_test_acc)))
    print('\n')
    print("dataname,learning_rate, weight_decay, nhid,num_scatter,epochs, dropout_prob",\
          dataname,learning_rate, weight_decay, nhid,num_scatter,epochs, dropout_prob)
    print(
        '***************************************************************************************************************************')
    return np.mean(saved_model_test_acc)

def ray_train(args):
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    class ExperimentTerminationReporter(CLIReporter):
        def should_report(self, trials, done=True):
            """Reports only on experiment termination."""
            return done

    def training_function(config):
        acc = objective(args,**config)
        tune.report(acc=acc)

    # ray.shutdown()
    # ray.init(ignore_reinit_error=True)
    ray.init(address='auto')

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
            "learning_rate": tune.grid_search([1e-6,5e-6]),
            "dataname": tune.grid_search(["ogbn-arxiv"]),#"cora","PubMed","Citeseer""CS","wikics","ogbn-arxiv","wisconsin","texas"]
            "weight_decay": tune.grid_search([1e-4,1e-5]),
            "epochs": tune.grid_search([500,200]),
            "if_linear": tune.grid_search([True,False]),
            "dropout_prob": tune.grid_search([0,0.1]),
            "head": tune.grid_search([2,4]),####no gat no head
            "nhid": tune.grid_search([64,256]),
            "num_scatter": tune.grid_search([3]),
        },
        progress_reporter=ExperimentTerminationReporter(),
        resources_per_trial={'gpu': 1, 'cpu': 2},
        # scheduler=asha_scheduler
    )

    print("Best config: ", analysis.get_best_config(
        metric="acc", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    ray.shutdown()
    return df
def arg_train(args):
    df = objective(args,learning_rate=args.lr, weight_decay=args.wd, nhid=args.nhid,epochs=args.epochs, dataname=args.dataset)

if __name__ == '__main__':
    # get config
    parser.add_argument("action", type=str, default=ray_train, help="ray or arg train")
    parser.add_argument('--dataset', type=str, default='wisconsin',
                        help='name of dataset with choices "Cora", "Citeseer", "Wikics"')
    parser.add_argument('--datatype', type=str, default='noisy',
                        help='data type with choices "clean", "noisy", "denoised"')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='data anomal ratio')
    parser.add_argument('--epochs', type=int, default=100,
                        help='')
    parser.add_argument('--nhid', type=int, default=128,
                        help='')
    parser.add_argument('--mettackdata', type=int, default=0,
                        help='use mettack noise data')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--wd', type=float, default=0.0001)
    ###airgnn args
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--lambda_amp', type=float, default=0.1)
    parser.add_argument('--lcc', type=str2bool, default=False)
    parser.add_argument('--normalize_features', type=str2bool, default=True)
    parser.add_argument('--random_splits', type=str2bool, default=False)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--reps', type=int, default=3)
    # parser.add_argument('--dropout', type=float, default=0.8, help="dropout")
    parser.add_argument('--K', type=int, default=10, help="the number of propagagtion in AirGNN")
    parser.add_argument('--model_cache', type=str2bool, default=False)
    args = parser.parse_args()
    if args.action=="arg_train":
        arg_train(args)
    if args.action=="ray_train":
        ray_train(args)