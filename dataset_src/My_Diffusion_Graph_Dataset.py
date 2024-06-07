import torch
import torch.nn.functional as F
import scipy.io
from torch_geometric.data import InMemoryDataset, download_url, Data
from scipy import sparse
import numpy as np
from torch_geometric.utils import get_laplacian, degree
from scipy.sparse import csr_matrix
import scipy.sparse as sp
def adj_torch_sparse(A):
    ed = sp.coo_matrix(A)
    indices = np.vstack((ed.row, ed.col))
    index = torch.LongTensor(indices)
    values = torch.Tensor(ed.data)
    torch_sparse_mat = torch.sparse_coo_tensor(index, values, ed.shape)
    return torch_sparse_mat

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
def diffusion_scat(edge_index, K=3,diffusion="randomwalk"):
    scatter_index = []
    scatter_value = []
    num_nodes = edge_index.max().item() + 1
    np_edge = edge_index.cpu().numpy()
    row, col = np_edge[0], np_edge[1]
    adj_value = np.ones_like(row)
    W = csr_matrix((adj_value, (row, col)), shape=(num_nodes, num_nodes))
    D = degree(edge_index[0], num_nodes).cpu().numpy()
    if np.min(D)==0:
        D = np.where(D>0,D,1)
        print("D:",D)
    Dhalf = np.diag(1 / np.sqrt(D))
    A = np.matmul(np.matmul(Dhalf, W.todense()), Dhalf)
    if diffusion == "sym":
        T = (np.eye(np.shape(D)[0]) + A) / 2  ###(I+T)/2,sym diffusion
    else:
        T = (np.eye(np.shape(D)[0]) + np.matmul(W.todense(), np.diag(1 / D))) / 2
    t = 2^(K-1)
    U = np.linalg.matrix_power(T, t)  ###U=T^3(2^2-1=3)
    #U = np.eye(num_nodes) / num_nodes
    U = adj_torch_sparse(U)
    U_index = U.coalesce().indices()
    U_value = U.coalesce().values()
    scatter_index.append(U_index)
    scatter_value.append(U_value)
    for idx in range(K):
        if idx == 0:
            tmppsi = adj_torch_sparse(np.eye(np.shape(D)[0]) - T)
            tmppsi_index = tmppsi.coalesce().indices()
            tmppsi_value = tmppsi.coalesce().values()
            scatter_index.append(tmppsi_index)
            scatter_value.append(tmppsi_value)
        else:
            T0 = T
            T = np.matmul(T0, T0)
            tmppsi = adj_torch_sparse(T0 - T)
            tmppsi_index = tmppsi.coalesce().indices()
            tmppsi_value = tmppsi.coalesce().values()
            scatter_index.append(tmppsi_index)
            scatter_value.append(tmppsi_value)
    return scatter_index,scatter_value
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
            #scatter_rep = getRep(x,lamb,V,layer=3)##(2**layer-1)*num_nodes
            scatter_index, scatter_value = diffusion_scat(edge_index)
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]]

            y_origin = target[i].item()
            y = (target[i].item() - mean) / std
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, scatter_index=scatter_index,scatter_value=scatter_value)
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
            data1 = Data(x=dataset[i].x, edge_index=dataset[i].edge_index, y=dataset[i].y,scatter_index=dataset[i].scatter_index,scatter_value=dataset[i].scatter_value)####pyG dataset
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
from typing import Optional, Callable, List
import os
import os.path as osp
import shutil
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
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
        self.data, self.slices ,self.sizes = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

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
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels  ###protein=4-3

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

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
    #     self.name = name
    #     self.cleaned = cleaned
    #     super().__init__(root, transform, pre_transform, pre_filter)
    #
    #     out = torch.load(self.processed_paths[0])
    #     if not isinstance(out, tuple) and len(out) != 3:
    #         raise RuntimeError(
    #             "The 'data' object was created by an older version of PyG. "
    #             "If this error occurred while loading an already existing "
    #             "dataset, remove the 'processed/' directory in the dataset's "
    #             "root folder and try again.")
    #     self.data, self.slices, self.sizes = out
    #
    # @property
    # def raw_dir(self) -> str:
    #     name = f'raw{"_cleaned" if self.cleaned else ""}'
    #     return osp.join(self.root, self.name, name)
    #
    # @property
    # def processed_dir(self) -> str:
    #     name = f'processed{"_cleaned" if self.cleaned else ""}'
    #     return osp.join(self.root, self.name, name)
    #
    # @property
    # def num_node_labels(self) -> int:
    #     return self.sizes['num_node_labels']
    #
    # @property
    # def num_node_attributes(self) -> int:
    #     return self.sizes['num_node_attributes']
    #
    # @property
    # def num_edge_labels(self) -> int:
    #     return self.sizes['num_edge_labels']
    #
    # @property
    # def num_edge_attributes(self) -> int:
    #     return self.sizes['num_edge_attributes']
    #
    # @property
    # def raw_file_names(self) -> List[str]:
    #     names = ['A', 'graph_indicator']
    #     return [f'{self.name}_{name}.txt' for name in names]
    #
    # @property
    # def processed_file_names(self) -> str:
    #     return 'data.pt'
    #
    # def download(self):
    #     url = self.cleaned_url if self.cleaned else self.url
    #     folder = osp.join(self.root, self.name)
    #     path = download_url(f'{url}/{self.name}.zip', folder)
    #     extract_zip(path, folder)
    #     os.unlink(path)
    #     shutil.rmtree(self.raw_dir)
    #     os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)

        data_list = [self.get(idx) for idx in range(len(self))]
        new_data_list = []
        for data in data_list:
            edge_index = data.edge_index
            x_in = data.x
            scatter_index, scatter_value = diffusion_scat(edge_index)
            # print("max:",np.max(scatter_list[0]),np.max(scatter_list[1]),np.max(scatter_list[2]),)
            data.scatter_index = scatter_index  ### = Data(x= x_in,edge_index=edge_index, y=data.y,scatter_list=scatter_list)
            data.scatter_value = scatter_value
            new_data_list.append(data)

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            if self.pre_filter is not None:
                data_list = [self.pre_filter(d) for d in data_list]
                for data in data_list:
                    edge_index = data.edge_index
                    x_in = data.x
                    # scatter_rep = getRep(x_in, lamb, V, layer=3)  ##(2**layer-1)*num_nodes
                    scatter_index, scatter_value = diffusion_scat(edge_index)
                    # print("max:",np.max(scatter_list[0]),np.max(scatter_list[1]),np.max(scatter_list[2]),)
                    data.scatter_index = scatter_index  ### = Data(x= x_in,edge_index=edge_index, y=data.y,scatter_list=scatter_list)
                    data.scatter_value = scatter_value
                    new_data_list.append(data)

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
                new_data_list=[]
                for data in data_list:
                    edge_index = data.edge_index
                    #scatter_rep = getRep(x_in, lamb, V, layer=3)  ##(2**layer-1)*num_nodes
                    scatter_index, scatter_value = diffusion_scat(edge_index)
                    # print("max:",np.max(scatter_list[0]),np.max(scatter_list[1]),np.max(scatter_list[2]),)
                    data.scatter_index = scatter_index  ### = Data(x= x_in,edge_index=edge_index, y=data.y,scatter_list=scatter_list)
                    data.scatter_value = scatter_value
                    new_data_list.append(data)
        self.data, self.slices = self.collate(new_data_list)
        torch.save((self.data, self.slices, sizes), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg  #read_csv_graph_pyg
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
            print("shape x_in:", x_in.shape,data.y.shape,data.y)
            scatter_index,scatter_value = diffusion_scat(edge_index)
            # print("max:",np.max(scatter_list[0]),np.max(scatter_list[1]),np.max(scatter_list[2]),)
            data.scatter_index = scatter_index  ### = Data(x= x_in,edge_index=edge_index, y=data.y,scatter_list=scatter_list)
            data.scatter_value = scatter_value
            data.y = (data.y).squeeze()
            new_data_list.append(data)

        data, slices = self.collate(new_data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
