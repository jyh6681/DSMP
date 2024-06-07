
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import get_laplacian
from typing import Optional, Callable, List
import os
import os.path as osp
import shutil
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
import matplotlib.pyplot as plt
import seaborn as sns
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg  #read_csv_graph_pyg
import pandas as pd
import shutil, os
import os.path as osp
import scipy
thres = 1.2

def plot_freq(eigen_list,name):
    less_than_half = 0
    total = len(eigen_list)
    for d in eigen_list:
        if d < thres:
            less_than_half += 1      
    low_ratio = less_than_half / total
    high_ratio = round(1-low_ratio,3)
    bar_width = 0.2
    bins = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,2.2]
    counts, bins = np.histogram(eigen_list, bins=bins,)
    colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000', 
        '#0000FF', '#87CEEB', '#4169E1', '#9932CC', 
        '#2E5894', '#8A2BE2', '#A52A2A', '#DEB887', '#D2691E']
    for i, c in enumerate(counts):
        plt.bar(bins[i:i+1], c, color=colors[i],alpha=0.5,width=bar_width)
    # 添加标签
    plt.xlabel('Frequency Range',fontsize=12)
    plt.ylabel('Count',fontsize=12)
    # plt.text(1, 1, 'Ratio = '+str(high_ratio), fontsize=12, color='red')
    plt.text(0.95, 0.95, 'Ratio = '+str(high_ratio), fontsize=12, ha='right', va='top', transform=plt.gca().transAxes)
    # plt.title('High Frequency Ratio = '+str(high_ratio),fontsize=5)

    # 保存图片
    path = "/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/data/diffusion_data4.9/"
    plt.savefig(path+str(name)+'_'+ 'histogram_'+str(thres)+'.png',bbox_inches='tight', dpi=800)
    plt.close()

def plot_gap(gap_list,name):
    bar_width = 0.2
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    counts, bins = np.histogram(gap_list, bins=bins,)
    colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000', 
        '#0000FF', '#87CEEB', '#4169E1', '#9932CC', 
        '#2E5894', '#8A2BE2', '#A52A2A', '#DEB887', '#D2691E']
    for i, c in enumerate(counts):
        plt.bar(bins[i:i+1], c, color=colors[i],alpha=0.5,width=bar_width)
    # 添加标签
    plt.xlabel('Spectral Gap Range',fontsize=12)
    plt.ylabel('Count',fontsize=12)
    plt.text(0.95, 0.95, 'Mean = '+str(round(sum(gap_list)/len(gap_list),3)), fontsize=12, ha='right', va='top', transform=plt.gca().transAxes)

    # 保存图片
    path = "/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/data/diffusion_data4.9/"
    plt.savefig(path+str(name)+'_'+ 'gap.png',bbox_inches='tight', dpi=800)
    plt.close()

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
        return 'newdata.pt'

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
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]]
            y_origin = target[i].item()
            y = (target[i].item() - mean) / std
            data = Data(x= x, edge_index=edge_index, edge_attr=edge_attr)
            data.num_nodes = edge_index.max().item() + 1
            data.y_origin = y_origin
            data_list.append(data)

        eigen_list = []
        gap_list = []
        for data in data_list:
            edge_index = data.edge_index
            num_nodes =  torch.max(edge_index)+1 # data.x.shape[0]
            L = get_laplacian(edge_index, num_nodes=num_nodes, normalization='sym')
            L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))
            lamb, V = np.linalg.eigh(L.toarray())
            eigen_list.extend(lamb.tolist())
            gap_list.append(sorted(lamb.tolist())[1] )
        # 统计每个区间的数据量
        plot_freq(eigen_list,name="qm7")
        plot_gap(gap_list,name="qm7")



class EigenTUD(InMemoryDataset):
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
        self.root = root
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
        return 'newdata.pt'
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
        eigen_list = []
        gap_list = []
        for data in data_list:
            edge_index = data.edge_index
            num_nodes =  torch.max(edge_index)+1 # data.x.shape[0]
            L = get_laplacian(edge_index, num_nodes=num_nodes, normalization='sym')
            L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))
            lamb, V = np.linalg.eigh(L.toarray())
            eigen_list.extend(lamb.tolist())
            gap_list.append(sorted(lamb.tolist())[1] )
        # 统计每个区间的数据量
        plot_freq(eigen_list,name=self.name)
        plot_gap(gap_list,name=self.name)

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


class EigenGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root="dataset", transform=None, pre_transform=None):
        self.name = name  ## original name, e.g., ogbg-mol-tox21
        self.dir_name = "_".join(name.split("-")) + "_pyg"  ## replace hyphen with underline, e.g., ogbg_mol_tox21_pyg

        self.original_root = root
        self.root = osp.join(root, self.dir_name)

        self.meta_info = pd.read_csv(os.path.join("/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/src/dataset_src/", "master.csv"), index_col=0)
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

        super(EigenGraphPropPredDataset, self).__init__(self.root, transform, pre_transform)

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
        return "new_data_processed.pt"

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
        eigen_list = []
        gap_list = []
        for data in data_list:
            edge_index = data.edge_index
            num_nodes =  torch.max(edge_index)+1 # data.x.shape[0]
            L = get_laplacian(edge_index, num_nodes=num_nodes, normalization='sym')
            L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))
            lamb, V = np.linalg.eigh(L.toarray())
            eigen_list.extend(lamb.tolist())
            gap_list.append(sorted(lamb.tolist())[1] )
        # 统计每个区间的数据量
        plot_freq(eigen_list,name=self.name)
        plot_gap(gap_list,name=self.name)




dataname_list =["qm7", "NCI1", "COLLAB", "DD","ENZYMES","MUTAG","IMDB-BINARY","PROTEINS","ogbg-molhiv",]
# dataname_list = ["IMDB-BINARY"]# ["COLLAB","ogbg-molhiv","IMDB-BINARY"]
for dataname in dataname_list:
    try:
        path = osp.join("/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/", 'data', "diffusion_data4.9", dataname)
        if dataname=="ogbg-molhiv":
            dataset = EigenGraphPropPredDataset(root=path,name=dataname)
        if dataname=="qm7":
            dataset = SVDQM7("/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/data/diffusion_data4.9/")
        else:
            dataset = EigenTUD(path, name=dataname)
            
    except:
        pass

