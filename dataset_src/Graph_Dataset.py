import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
import scipy.io
from torch_geometric.data import InMemoryDataset, download_url, Data,extract_zip
from torch_geometric.utils import get_laplacian
from scipy import sparse
import math
from scipy.sparse.linalg import lobpcg
from numba import jit
import scipy.sparse as sp
import time
from typing import Optional, Callable, List
import shutil
from torch_geometric.io import read_tu_data
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg  #read_csv_graph_pyg
class QM7(InMemoryDataset):
    url = 'http://quantum-machine.org/data/qm7.mat'

    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(QM7, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'qm7.mat'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data = scipy.io.loadmat(self.raw_paths[0])
        coulomb_matrix = torch.from_numpy(data['X'])
        target = torch.from_numpy(data['T']).to(torch.float).squeeze(0)
        mean = torch.mean(target).item()
        std = torch.sqrt(torch.var(target)).item()

        data_list = []
        for i in range(target.shape[0]):
            edge_index = coulomb_matrix[i].nonzero().t().contiguous()####atom link, node  max num=23
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]]
            y_origin = target[i].item()
            y = (target[i].item() - mean) / std
            data = Data(edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.num_nodes = edge_index.max().item() + 1
            data.y_origin = y_origin
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


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


def prop_scatter_mat(lamb, V, J=4):  # to replace propLayer,V is the adjacent matrix
    y = []
    for k in range(J):
        j = J - k ##k=0,1,2:j=3,2,1
        if j == J:
            H = phi(2 ** j * lamb)###j=3
        else:
            H = psi(2 ** (-j) * lamb)##j=1,2
        H = np.diag(H)
        scatter = np.matmul(np.matmul(V, H), V.T)
        y.append(scatter)######wavelet transform
    return y


def propLayerHaar(x, lamb, V, J=3):  # to replace propLayer,V is the adjacent matrix
    y = []
    for k in range(J):
        j = J - k
        if j == J:
            H = phi(2 ** j * lamb)
        else:
            H = psi(2 ** (-j) * lamb)
        H = np.diag(H)
        y.append(np.matmul(np.matmul(np.matmul(V, H), V.T), x))######wavelet transform
    return y
def getRep(f, lamb, V, layer=3):
    y_out = []
    y_next = []
    y = propLayerHaar(f, lamb, V)
    y_out.append(y.pop(0))
    y_next.extend(y)
    for i in range(layer - 1):
        for k in range(len(y_next)):
            ftemp = y_next.pop(0)
            ftemp = np.absolute(ftemp)
            y = propLayerHaar(ftemp, lamb, V)
            y_out.append(y.pop(0))
            y_next.extend(y)
    y_out = np.concatenate(tuple(y_out), axis=1)  # use this to form a single matrix
    return y_out

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
    index = torch.LongTensor(indices)
    values = torch.Tensor(ed.data)
    torch_sparse_mat = torch.sparse_coo_tensor(index, values, ed.shape)
    return torch_sparse_mat

@jit
def adjConcat(a, b):
    '''
    concat two adj matrix along the diag
    '''
    lena = len(a)
    lenb = len(b)
    left = np.row_stack((a, np.zeros((lenb, lena))))  # get the left
    right = np.row_stack((np.zeros((lena, lenb)), b))  # get the right
    result = np.hstack((left, right))
    return result

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
            d_list.append(d[i, l].toarray())
    return d_list #W_{0,J},W_{1,1},W_{1,J} the other is removed


class SVDQM7(InMemoryDataset):
    url = 'http://quantum-machine.org/data/qm7.mat'

    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(SVDQM7, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'qm7.mat'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self,layer=2):
        data = scipy.io.loadmat(self.raw_paths[0])
        coulomb_matrix = torch.from_numpy(data['X'])
        target = torch.from_numpy(data['T']).to(torch.float).squeeze(0)
        mean = torch.mean(target).item()
        std = torch.sqrt(torch.var(target)).item()

        data_list = []
        for i in range(target.shape[0]):
            edge_index = coulomb_matrix[i].nonzero().t().contiguous()####atom link, node  max num=23
            ###SVD scatter
            num_nodes = edge_index.max().item() + 1
            x = torch.ones(num_nodes, 5)  ####all one node feat
            scatter_list = chebyshev_mat(edge_index) ####list
            scatter_index,scatter_value = scatter_index_value(scatter_list)
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]]
            y_origin = target[i].item()
            y = (target[i].item() - mean) / std
            data = Data(x= x, edge_index=edge_index, edge_attr=edge_attr, y=y,scatter_edge_index=scatter_index,scatter_edge_attr = scatter_value)
            data.num_nodes = edge_index.max().item() + 1
            data.y_origin = y_origin
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def MyDataset(dataset, num_features, QM7=True):
    dataset1 = list()
    label = list()
    for i in range(len(dataset)):
        if QM7:
            #x_qm7 = torch.ones(dataset[i].num_nodes, num_features)####all one node feat
            data1 = dataset[i]#Data(x=dataset[i].x, edge_index=dataset[i].edge_index, y=dataset[i].y,scatter_edge_index=dataset[i].scatter_edge_index,scatter_edge_attr = dataset[i].scatter_edge_attr)####pyG dataset
            data1.y_origin = dataset[i].y_origin
            label.append(dataset[i].y_origin)
        # append data1 into dataset1
        dataset1.append(data1)
    if QM7:
        mean = torch.mean(torch.Tensor(label)).item()
        std = torch.sqrt(torch.var(torch.Tensor(label))).item()
        return dataset1, mean, std
    else:
        return dataset1


def qm7_test(model, loader, device, mean, std):
    model.eval()
    loss = 0.
    for data in loader:
        data = data.to(device)
        out = model(data)
        out = out * std + mean
        loss += F.l1_loss(out, data.y_origin, reduction='sum').item()
    return loss / len(loader.dataset)


def qm7_test_train(model, loader, device):
    model.eval()
    loss = 0.
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss += F.mse_loss(out, data.y, reduction='sum').item()
    return loss / len(loader.dataset)

def qm_test(model, loader, device, mean, std,dataname="qm7"):
    model.eval()
    loss = 0.
    loss_all = torch.zeros((19)).to(device)
    for data in loader:
        data = data.to(device)
        out = model(data)
        out = out * std + mean
        if dataname=="qm7":
            loss += F.l1_loss(out, data.y_origin, reduction='sum').item()
        else:
            loss += F.l1_loss(out, data.y, reduction='sum').item()
            loss_all += torch.sum(torch.abs(out-data.y),dim=0)
    if dataname=="QM9":
        return loss / len(loader.dataset),(loss_all/ len(loader.dataset)).detach().cpu().numpy()
    else:
        return loss / len(loader.dataset)


def qm_test_train(model, loader, device,mean,std,dataname="qm7"):
    model.eval()
    loss = 0.
    for data in loader:
        data = data.to(device)
        out = model(data)
        if dataname == "qm7":
            loss += F.mse_loss(out, data.y, reduction='sum').item()
        else:
            loss += F.l1_loss(out, (data.y-mean)/std, reduction='sum').item()
    return loss/len(loader.dataset)


class SVDTUD(InMemoryDataset):
    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)

        out = torch.load(self.processed_paths[0])
        if not isinstance(out, tuple) and len(out) != 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        self.data, self.slices, self.sizes = out

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
        data_list = [self.get(idx) for idx in range(len(self))]
        new_data_list = []
        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            if self.pre_filter is not None:
                data_list = [self.pre_filter(d) for d in data_list]
                for data in data_list:
                    edge_index = data.edge_index
                    x_in = data.x
                    scatter_list = chebyshev_mat(edge_index) ####list
                    scatter_index,scatter_value = scatter_index_value(scatter_list)
                    data = Data(x=x_in, edge_index=edge_index, y=data.y, scatter_edge_index=scatter_index,scatter_edge_attr = scatter_value)
                    new_data_list.append(data)
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
                for data in data_list:
                    edge_index = data.edge_index
                    x_in = data.x
                    scatter_list = chebyshev_mat(edge_index) ####list
                    scatter_index,scatter_value = scatter_index_value(scatter_list)
                    data = Data(x=x_in, edge_index=edge_index, y=data.y, scatter_edge_index=scatter_index,scatter_edge_attr = scatter_value)
                    new_data_list.append(data)
            self.data, self.slices = self.collate(new_data_list)
            self._new_data_list = None  # Reset cache.
        else:
            for data in data_list:
                edge_index = data.edge_index
                x_in = data.x
                scatter_list = chebyshev_mat(edge_index) ####list
                scatter_index,scatter_value = scatter_index_value(scatter_list)
                data = Data(x=x_in, edge_index=edge_index, y=data.y, scatter_edge_index=scatter_index,scatter_edge_attr = scatter_value)
                new_data_list.append(data)
                self.data, self.slices = self.collate(new_data_list)
        data_list = [self.get(idx) for idx in range(len(self))]
        torch.save((self.data, self.slices, sizes), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


class SVDGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root="dataset", transform=None, pre_transform=None):
        self.name = name  ## original name, e.g., ogbg-mol-tox21
        self.dir_name = "_".join(name.split("-")) + "_pyg"  ## replace hyphen with underline, e.g., ogbg_mol_tox21_pyg

        self.original_root = root
        self.root = osp.join(root, self.dir_name)

        self.meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), "master.csv"), index_col=0)
        if not self.name in self.meta_info:
            print(self.name)
            error_mssg = "Invalid dataset name {}.\n".format(self.name)
            error_mssg += "Available datasets are as follows:\n"
            error_mssg += "\n".join(self.meta_info.keys())
            raise ValueError(error_mssg)

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (
        not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info[self.name]['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.root)

        self.download_name = self.meta_info[self.name]["download_name"]  ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info[self.name]["num tasks"])
        self.eval_metric = self.meta_info[self.name]["eval metric"]
        self.task_type = self.meta_info[self.name]["task type"]
        self.__num_classes__ = int(self.meta_info[self.name]["num classes"])

        super(SVDGraphPropPredDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info[self.name]["split"]

        path = osp.join(self.root, "split", split_type)

        train_idx = pd.read_csv(osp.join(path, "train.csv.gz"), compression="gzip", header=None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, "valid.csv.gz"), compression="gzip", header=None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, "test.csv.gz"), compression="gzip", header=None).values.T[0]

        return {"train": torch.tensor(train_idx, dtype=torch.long), "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long)}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        file_names = ["edge"]
        if self.meta_info[self.name]["has_node_attr"] == "True":
            file_names.append("node-feat")
        if self.meta_info[self.name]["has_edge_attr"] == "True":
            file_names.append("edge-feat")
        return [file_name + ".csv.gz" for file_name in file_names]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        url = self.meta_info[self.name]["url"]
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

        else:
            print("Stop downloading.")
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        ### read pyg graph list
        add_inverse_edge = self.meta_info[self.name]["add_inverse_edge"] == "True"

        if self.meta_info[self.name]["additional node files"] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info[self.name]["additional node files"].split(',')

        if self.meta_info[self.name]["additional edge files"] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info[self.name]["additional edge files"].split(',')

        data_list = read_graph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge,
                                       additional_node_files=additional_node_files,
                                       additional_edge_files=additional_edge_files)

        if self.task_type == "subtoken prediction":
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, "graph-label.csv.gz"), compression="gzip",
                                                header=None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            graph_label = pd.read_csv(osp.join(self.raw_dir, "graph-label.csv.gz"), compression="gzip",
                                      header=None).values

            has_nan = np.isnan(graph_label).any()

            for i, g in enumerate(data_list):
                if "classification" in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.long)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        ###scatter pretransform
        new_data_list = []
        for data in data_list:
            # print("data:", data,data.y,data.x)
            edge_index = data.edge_index
            x_in = data.x
            scatter_list = chebyshev_mat(edge_index) ####list
            scatter_index,scatter_value = scatter_index_value(scatter_list)
            data = Data(x=x_in, edge_index=edge_index, y=(data.y).squeeze(), scatter_edge_index=scatter_index,scatter_edge_attr = scatter_value)
            new_data_list.append(data)
        data, slices = self.collate(new_data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
