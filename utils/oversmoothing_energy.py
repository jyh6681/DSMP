import torch
from model.DSMP import *
from torch_geometric.utils import get_laplacian
from smp_graphlevel_noabs import *
import random
import seaborn as sns
# 计算Dirichlet能量
def compute_dirichlet_energy(edge_index,output):
    # 计算每一层输出结果的Dirichlet能量
    energies = []
    for i in range(len(output)):
        x = output[i]
        laplacian = compute_laplacian(edge_index, x.size(0))
        energy =  torch.trace(torch.matmul(x.t(), laplacian).matmul(x))
        energies.append(energy.item())
    return energies
def compute_laplacian(edge_index, num_nodes):
    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=torch.float).to(device)
    deg.index_add_(0, row, torch.ones_like(row, dtype=torch.float).to(device))
    deg_inv_sqrt = deg.pow(-0.5)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    laplacian = torch.sparse_coo_tensor(edge_index, norm, (num_nodes, num_nodes))
    return laplacian.to_dense()

class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, dropout_prob, hid_fc_dim, pRatio, head=1,layer=2,num_scatter=3,if_gat=True,if_linear=True):
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
        self.convs = nn.ModuleList([])
        for i in range(num_scatter):#######each scatt layer is two step of scattering say it is MS and here we use num_scatter MS form the Net
            self.convs.append(Scatter(in_channels=nhid, out_channels=nhid,dropout_prob=dropout_prob, if_gat=if_gat ,head=head,layer=layer))
        # hidden_dim = [nhid] + hid_fc_dim + [num_classes]
        # self.global_add_pool = global_add_pool
        # fcList = [score_block(i, j, self.dropout_prob) for i, j in zip(hidden_dim[:-2], hidden_dim[1:])]
        # fcList.append(nn.Sequential(nn.Linear(*hidden_dim[-2:]),nn.BatchNorm1d(hidden_dim[-1])))
        # self.fc = nn.Sequential(*fcList)
    def forward(self, data):
        x_batch, edge_index_batch, batch,scatter_edge_index,scatter_edge_attr = (data.x).float(), data.edge_index, data.batch,data.scatter_edge_index,data.scatter_edge_attr
        _, graph_size = torch.unique(batch, return_counts=True)
        if self.if_linear == True:
            x_batch = self.lin_in(x_batch)
        hidden_rep = x_batch
        out = []
        scatter_list =[]
        for index in range(self.num_scatter):
            hidden_rep += self.convs[index](hidden_rep,scatter_edge_index,scatter_edge_attr,edge_index_batch)
            out.append(hidden_rep.clone())
            scatter_list.append(self.convs[index](hidden_rep,scatter_edge_index,scatter_edge_attr,edge_index_batch))
        # h_pooled = self.global_add_pool(hidden_rep, batch)#grasspool(hidden_rep, graph_size, self.pRatio)
        # x = self.fc(h_pooled)
        return scatter_list,out

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList([
            GCNConv(in_channels, hidden_channels)
        ])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, data):
        x, edge_index = (data.x).float(), data.edge_index
        out = []
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            out.append(x)
        x = self.convs[self.num_layers - 1](x, edge_index)
        return x,out

class GAT(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels, out_channels, num_layers, num_heads=1):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList([
            GATConv(in_channels, hidden_channels,heads=num_heads)
        ])
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels,heads=num_heads))
        self.convs.append(GATConv(hidden_channels, out_channels,heads=num_heads))

    def forward(self, data):
        x, edge_index = (data.x).float(), data.edge_index
        out = []
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            out.append(x)
        x = self.convs[self.num_layers - 1](x, edge_index)
        return x,out


seeds = [0,1024,3389,44]

for seed in seeds:
    random.seed(seed)
    device = "cuda"
    dataname = "PROTEINS"
    path = osp.join("/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/", 'data', "diffusion_data4.9", dataname)
        
    dataset = SVDTUD(path, name=dataname) ###########################proteins remove first col
    num_train, num_val = int(len(dataset) * 0.8), int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_train + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_train, num_val, num_test])
    train_loader = DataLoader(training_set, batch_size=1, shuffle=False)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    loss_criteria = F.cross_entropy
    num_features = dataset.num_features
    print("features:", num_features)
    num_classes = dataset.num_classes
    print("num_calsses:", num_classes)
    # 计算Dirichlet能量
    layers=50
    nhid = 64
    fc_dim=[64, 16]
    dropout_prob = 0.1
    pRatio = 0.5
    dsmp_model = Net(num_features, nhid, num_classes, dropout_prob, fc_dim, pRatio, num_scatter=layers,head=1,if_gat=True,
                        if_linear=False).to(device)
    gcn_model = GCN(num_features, nhid, num_classes,layers).to(device)
    gat_model = GAT(num_features, nhid, num_classes,layers).to(device)
    for data in train_loader:
        data = data.to(device)
        scatter_list,dsmp_out = dsmp_model(data)
        _,gcn_out = gcn_model(data)
        _,gat_out = gat_model(data)
        print("dsmp_out")
        dsmp_energies = compute_dirichlet_energy(data.edge_index,dsmp_out)
        print("dsmp energy:",dsmp_energies)
        gcn_energies = compute_dirichlet_energy(data.edge_index,gcn_out)
        gat_energies = compute_dirichlet_energy(data.edge_index,gat_out)
        scatter_energies = compute_dirichlet_energy(data.edge_index,scatter_list)
        break

    #打印原始信号的Dirichlet能量
    for data in train_loader:
        data = data.to(device)
        laplacian = compute_laplacian(data.edge_index, data.x.size(0))
        energy_origin = torch.trace(torch.matmul(data.x.t(), laplacian).matmul(data.x)).item()
        break
    dsmp_energies.insert(0,energy_origin)
    gcn_energies.insert(0,energy_origin)
    gat_energies.insert(0,energy_origin)


    print("sca:",scatter_energies)
    # colors = ['#FBB4AE', '#B3CDE3', '#CCEBC5']
    colors = [(0.2, 0.4, 0.6),(210/255, 180/255, 140/255),(139/255, 0, 0)]#['#E8A628', '#B87D4B', '#6B4226']
    # colors = ['#E8A628', '#B87D4B', '#6B4226']
    #颜色1: #440154,颜色2: #31688E,颜色3: #35B779
    plt.plot(range(0, len(dsmp_energies)), dsmp_energies,color=colors[0], label="DSMP")
    plt.plot(range(0, len(gcn_energies)), gcn_energies,color=colors[1], label="GCN")
    plt.plot(range(0, len(gat_energies)), gat_energies, color=colors[2],label="GAT")
    # plt.plot(range(0, len(gat_energies)-1), scatter_energies, color="#6B4226",label="scatter")
    # 设置横坐标刻度显示为整数
    # plt.xticks(range(len(gcn_energies)), map(int, range(0, len(gcn_energies))))
    plt.xlabel('Layers')
    plt.ylabel('Dirichlet Energy')
    plt.legend()
    plt.show()
    plt.savefig("/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/show_result/over_smooth_seed"+str(seed)+"_sns.png",dpi=500)
    plt.close()