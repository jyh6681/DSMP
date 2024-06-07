import torch
from model.DSMP import *
from torch_geometric.utils import get_laplacian
from smp_graphlevel_noabs import *
import random
import statistics
import time
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
            # print("using linear transform")
            self.lin_in = Linear(num_features, nhid)
        else:
            # print("without using linear transform")
            nhid=num_features
        self.convs = nn.ModuleList([])
        for i in range(num_scatter):#######each scatt layer is two step of scattering say it is MS and here we use num_scatter MS form the Net
            self.convs.append(Scatter(in_channels=nhid, out_channels=nhid,dropout_prob=dropout_prob, if_gat=if_gat ,head=head,layer=layer))
        hidden_dim = [nhid] + hid_fc_dim + [num_classes]
        self.global_add_pool = global_add_pool
        fcList = [score_block(i, j, self.dropout_prob) for i, j in zip(hidden_dim[:-2], hidden_dim[1:])]
        fcList.append(nn.Sequential(nn.Linear(*hidden_dim[-2:]),nn.BatchNorm1d(hidden_dim[-1])))
        self.fc = nn.Sequential(*fcList)
    def forward(self, data):
        x_batch, edge_index_batch, batch,scatter_edge_index,scatter_edge_attr = (data.x).float(), data.edge_index, data.batch,data.scatter_edge_index,data.scatter_edge_attr
        _, graph_size = torch.unique(batch, return_counts=True)
        if self.if_linear == True:
            x_batch = self.lin_in(x_batch)
        hidden_rep = x_batch
        out = []
        for index in range(self.num_scatter):
            hidden_rep += self.convs[index](hidden_rep,scatter_edge_index,scatter_edge_attr,edge_index_batch)
            out.append(hidden_rep.clone())
        h_pooled = self.global_add_pool(hidden_rep, batch)#grasspool(hidden_rep, graph_size, self.pRatio)
        x = self.fc(h_pooled)
        if self.num_classes == 1:
            return x.view(-1)
        else:
            return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.convs = torch.nn.ModuleList([
            GCNConv(in_channels, hidden_channels)
        ])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.global_add_pool = global_add_pool
    def forward(self, data):
        x, edge_index = (data.x).float(), data.edge_index
        out = []
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            out.append(x)
        x = self.convs[self.num_layers - 1](x, edge_index)
        x = self.global_add_pool(x, data.batch)#grasspool(hidden_rep, graph_size, self.pRatio)
        if self.out_channels == 1:
            return x.view(-1)
        else:
            return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels, out_channels, num_layers, num_heads=1):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.convs = torch.nn.ModuleList([
            GATConv(in_channels, hidden_channels,heads=num_heads)
        ])
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels,heads=num_heads))
        self.convs.append(GATConv(hidden_channels, out_channels,heads=num_heads))
        self.global_add_pool = global_add_pool
    def forward(self, data):
        x, edge_index = (data.x).float(), data.edge_index
        out = []
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            out.append(x)
        x = self.convs[self.num_layers - 1](x, edge_index)
        x = self.global_add_pool(x, data.batch)#grasspool(hidden_rep, graph_size, self.pRatio)
        if self.out_channels == 1:
            return x.view(-1)
        else:
            return x

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.convs = torch.nn.ModuleList([
            GINConv(in_channels, hidden_channels)
        ])
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(hidden_channels, hidden_channels))
        self.convs.append(GINConv(hidden_channels, out_channels))
        self.global_add_pool = global_add_pool
    def forward(self, data):
        x, edge_index = (data.x).float(), data.edge_index
        out = []
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            out.append(x)
        x = self.convs[self.num_layers - 1](x, edge_index)
        x = self.global_add_pool(x, data.batch)#grasspool(hidden_rep, graph_size, self.pRatio)
        if self.out_channels == 1:
            return x.view(-1)
        else:
            return x



parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='ogbg-molhiv',
                    help='name of dataset (default: PROTEINS)')
parser.add_argument('--single_data', action='store_true')
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
parser.add_argument('--drop', type=float, default=0.0,
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
parser.add_argument('-a','--appendix', type=str, default="10.19")

args = parser.parse_args()
learning_rate=args.lr
weight_decay=args.wd
nhid=args.nhid
epochs=args.epochs
dropout_prob=args.drop
batch_size=args.batch_size
if_linear=args.if_linear
if_gat=args.if_gat,
dataname=args.dataname
fc_dim=[64, 16]
pRatio = 0.5
num_reps = 3
head = 4
patience = 20



seed = args.seed#0#1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = "cuda"
path = osp.join("/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/", 'data', "diffusion_data4.9", dataname)
    
dataset = SVDTUD(path, name=dataname) ###########################proteins remove first col
num_train, num_val = int(len(dataset) * 0.8), int(len(dataset) * 0.1)
num_test = len(dataset) - (num_train + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_train, num_val, num_test])
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
loss_criteria = F.cross_entropy
num_features = dataset.num_features
# print("features:", num_features)
num_classes = dataset.num_classes
# print("num_calsses:", num_classes)
# 计算Dirichlet能量
layer_list = [1,2,3,4,5,10,15,20,25,30,35,40,45,50,100]
model_names = ["gcn","gat"]#["dsmp",,"gin"]
for model_name in model_names:
    globals()[model_name+"_acc_list"] = []
    globals()[model_name+"_var_list"] = []

    for layers in layer_list:
        # create results matrix
        epoch_train_loss = np.zeros((num_reps, epochs))
        epoch_train_acc = np.zeros((num_reps, epochs))
        epoch_valid_loss = np.zeros((num_reps, epochs))
        epoch_valid_acc = np.zeros((num_reps, epochs))
        epoch_test_loss = np.zeros((num_reps, epochs))
        epoch_test_acc = np.zeros((num_reps, epochs))
        saved_model_loss = np.zeros(num_reps)
        saved_model_acc = np.zeros(num_reps)

        save_dir = "/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/smp_noabs_results_multilayers/"
        os.makedirs(save_dir,exist_ok=True)
        # training
        reps_acc = []
        for r in range(num_reps):
            if dataname=="ogbg-molhiv":
                split_idx = dataset.get_idx_split()
                training_set, validation_set, test_set = dataset[split_idx["train"]],dataset[split_idx["valid"]],dataset[split_idx["test"]]
            if model_name=="dsmp":
                model = Net(num_features, nhid, num_classes, dropout_prob, fc_dim, pRatio, num_scatter=layers,head=head,if_gat=if_gat,
                            if_linear=if_linear).to(device)
            if model_name=="gcn":
                model = GCN(num_features, nhid, num_classes,layers).to(device)
            if model_name=="gat":
                model = GAT(num_features, nhid, num_classes,layers).to(device)
            if model_name=="gin":
                model = GAT(num_features, nhid, num_classes,layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=3,verbose=False)
            early_stopping = EarlyStopping(patience=patience, verbose=False,path=save_dir+ str(dataname) + \
                                                "if_gat" + str(if_gat) +"drop"+str(dropout_prob)+ "lr"+str(learning_rate)+"wd"+str(weight_decay)+"heads"+str(head)+str(args.appendix) + '.pth')
            # start training
            min_loss = 1e10
            best_acc = 0
            best_dict = model.state_dict()
            # print("****** Start Rep {}: Training start ******".format(r + 1))
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
                    start = time.time()
                    train_acc, train_loss,auc = test(model, train_loader, device,dataname)
                    # if epoch==0 and r==0:
                    #     print(model_name," layers:",layers, "cost:", str(time.time()-start))
                    val_acc, val_loss,auc = test(model, val_loader, device,dataname)
                    test_acc, test_loss,auc = test(model, test_loader, device,dataname)
                    if best_acc<val_acc:
                        best_dict = model.state_dict()
                        best_acc=val_acc
                    epoch_train_acc[r, epoch], epoch_valid_acc[r, epoch], epoch_test_acc[r, epoch] = train_acc, val_acc, test_acc
                # if epoch%10==0:
                #     print(dataname+" on Epoch {}: Training accuracy: {:.4f}; Validation accuracy: {:.4f}; Test accuracy: {:.4f};Test auc: {:.4f}".format(epoch + 1, train_acc, val_acc, test_acc,auc))
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    # print("Early stopping \n")
                    break
                scheduler.step(val_loss)

                epoch_train_loss[r, epoch] = train_loss
                epoch_valid_loss[r, epoch] = val_loss
                epoch_test_loss[r, epoch] = test_loss
                # Test
                # print("****** Test start ******")
                # test_acc, test_loss,auc = test(model, test_loader, device,dataname)
                # if best_acc<test_acc:
                #     best_acc = test_acc
            
            model.load_state_dict(best_dict)
            test_acc, test_loss,auc = test(model, test_loader, device,dataname)
            saved_model_acc[r] = test_acc
            # print("args: ",args)
            # print("Test accuracy: {:.4f}".format(test_acc))
            reps_acc.append(test_acc)
            if dataname=="ogbg-molhiv":
                print("test auc:",auc)
        
        globals()[model_name+"_acc_list"].append(round(statistics.mean(reps_acc),4))
        globals()[model_name+"_var_list"].append(round(statistics.variance(reps_acc),4))
    print(model_name," acc/var:::::::",globals()[model_name+"_acc_list"],"\n",globals()[model_name+"_var_list"])