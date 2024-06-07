import torch
import pickle
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
from torch_geometric.data import Data
import os
import numpy as np
import scipy.sparse as sp
class HetroDataSet(InMemoryDataset):
    def __init__(self,root,name,transform=None,pre_transform=None):
        self.name = name
        self.root = root
        super(HetroDataSet, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['tcm_dataset.pt',]

    @property
    def processed_file_names(self):
        return ['tcm_dataset.pt',]

    def download(self):
        pass

    def process(self):

        # do processing, get x, y, edge_index ready.   
        graph_adjacency_list_file_path = os.path.join(self.root, self.name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join(self.root, self.name, f'out1_node_feature_label.txt')
        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))
        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        sparse_adj = sp.coo_matrix(adj)
        values = sparse_adj.data
        indices = np.vstack((sparse_adj.row, sparse_adj.col))  # 我们真正需要的coo形式
        edge_index = torch.LongTensor(indices)
        feat_mat = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
        num_class = len(np.unique(labels))
        graph = Data(x=torch.from_numpy(feat_mat), y=torch.from_numpy(labels), edge_index=edge_index)
        ####train val test split
        splits_file_path = self.root+"/splits/" + str(self.name) + "_split_0.6_0.2_" + str(1) + ".npz"
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
        graph.train_mask = torch.BoolTensor(train_mask)
        graph.val_mask = torch.BoolTensor(val_mask)
        graph.test_mask = torch.BoolTensor(test_mask)

        if self.pre_filter is not None:
            graph = [data for data in graph if self.pre_filter(data)]

        if self.pre_transform is not None:
            graph = [self.pre_transform(data) for data in graph]

        data, slices = self.collate([graph])
        torch.save((data, slices), self.processed_paths[0])
        
        