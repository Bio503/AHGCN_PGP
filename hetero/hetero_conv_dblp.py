import os
import os.path as osp
import scipy.sparse as sp
from DBN import DBN
import pandas as pd
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HeteroConv, Linear, SAGEConv

#path = osp.join(osp.dirname(osp.realpath(__file__)), '..\..\data\DBLP')
## We initialize conference node features with a single one-vector as feature:
#dataset = DBLP(path, transform=T.Constant())
#data = dataset[0]
#print(data)
def get_data(file_name=None):
    path = '../../data/bio'

    data = HeteroData()
    #读取节点特征
    G = torch.tensor(np.load(path + '/gene_feature_combined.npy'))
    data['G'].x = G
    D = torch.tensor(np.load(path + '/disease_feature_combined.npy'))
    data['D'].x = D
    #读取边index
    #read D D edge matrix from csv
    DDS = pd.read_csv(path + '/DDS.csv', index_col=0).to_numpy()
    DDS = torch.tensor(DDS.nonzero(),dtype=torch.long)
    data['D', 'DD', 'D'].edge_index = DDS
    
    #read G G edge matrix from csv
    GGS = pd.read_csv(path + '/GGS.csv', index_col=0).to_numpy()
    GGS = torch.tensor(GGS.nonzero(),dtype=torch.long)
    data['G', 'GG', 'G'].edge_index = GGS
    
    #read G D edge matrix from csv
    GD = pd.read_csv(path + f'/X_{file_name}.txt',header=None)
    #replace 'G' with'' and 'D' with ''
    GD = GD.replace('G', '', regex=True).replace('D', '', regex=True).to_numpy().astype(int).T
    data['G', 'GD', 'D'].edge_index = torch.tensor(GD-1,dtype=torch.long)
    
    #read G D edge matrix from csv
    #this is y
    data.y = torch.tensor(pd.read_csv(path + f'/y_{file_name}.txt',header=None).to_numpy().reshape(-1),dtype=torch.float)

    # PyTorch tensor functionality:
    # data = data.pin_memory()
    # data = data.to('cuda:0', non_blocking=True)
    return data

data_train = get_data("train")
data_test = get_data("test")

# GCN
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.Glin = Linear(hidden_channels, out_channels)
        self.Dlin = Linear(hidden_channels, out_channels)
        self.DBN = DBN(visible_units=128, hidden_units=[128,128,128], k=5,use_gpu=False)
        # self.DBN = DBN(visible_units=128, hidden_units=[128,128,128], k=5,use_gpu=True)
        
    def forward_GCN(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        #这里确定返回值
        G,D = self.Glin(x_dict['G']),self.Dlin(x_dict['D'])
        return G,D
    # 覆写父类nn.Module中的forward。
    # DBN=True调用DBN模型。
    def forward(self, x_dict, edge_index_dict, y=data_train.y[:1217], DBN = False):
        G,D = self.forward_GCN(x_dict, edge_index_dict)
        if DBN:
            G_num = G.shape[0]
            GD_node = torch.cat([G,D],dim=0)
            _,GD_node = self.DBN.train_static(GD_node,y)
            #split GD_node to G and D
            G,D = GD_node[:G_num],GD_node[G_num:]
        GD = G@(D.transpose(0,1))   # 1000*217
        #change GD_mask from coo to dense tensor
        GD_mask = torch.sparse_coo_tensor(data_train.edge_index_dict[('G', 'GD', 'D')], torch.ones_like(data_train.y), size=GD.shape).to_dense()
        GD = GD[GD_mask.bool()]
        print("GD=",GD)
        return GD

# 模型train\test
model = HeteroGNN(data_train.metadata(), hidden_channels=192, out_channels=128,
                  num_layers=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data, model = data.to(device), model.to(device)
data_train, model, data_test = data_train.to(device), model.to(device), data_test.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data_train.x_dict, data_train.edge_index_dict).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

# DBN为bool值
def train(DBN):
    model.train()
    optimizer.zero_grad()
    #x_dict:{节点类型：节点特征}
    #edge_index_dict:{(节点类型1，边类型，节点类型2)：边索引：[2,边数]}

    if DBN:
        out = model(data_train.x_dict, data_train.edge_index_dict, y=None, DBN = DBN)
        loss = F.cross_entropy(out, data_train.y)
    else:
        out = model(data_train.x_dict, data_train.edge_index_dict, DBN = DBN)
        loss = F.cross_entropy(out, data_train.y)
        loss.backward()
        optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data_test.x_dict, data_test.edge_index_dict)
    threshold = 0.5
    pred = np.mean(1*(pred>=threshold).numpy())     # mean()函数功能：求取均值 
    return pred

# 训练到一定阶段，接DBN
DBN_start = False
for epoch in range(1, 10000):
    loss = train(DBN = DBN_start)
    test_acc = test()
    # if test_acc>0.8:
    #     DBN_start = True
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
