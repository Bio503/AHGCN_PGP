import os.path as osp
import scipy.sparse as sp
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from torch_geometric.data import HeteroData
from DBN import DBN
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv, Linear, SAGEConv
from sklearn.metrics import roc_auc_score  # AUROC. Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve   # AUPRC
from sklearn.metrics import auc
# from sklearn.metrics import accuracy_score   #计算accuracy值
# from sklearn.metrics import roc_curve     # 画图用
from sklearn.metrics import recall_score

from decoder import InnerProductDecoder

# We initialize conference node features with a single one-vector as feature:

def get_train_data(file_name):
    # path = 'bio'
    path = "data_evalution"
    # path = "ad_data"

    data = HeteroData()
    # 1 读取节点特征
    # G = torch.tensor(np.load(path + '/g_ppi_feature_node2vec.npy'))
    # # G = torch.tensor(np.load(path + '/gene_feature_combined.npy'))
    # data['G'].x = G
    # D = torch.tensor(np.load(path + '/d_hpo_feature_node2vec.npy'))
    # # D = torch.tensor(np.load(path + '/disease_feature_combined.npy'))
    # data['D'].x = D

    #2 对比实验，不输入节点特征
    G = torch.tensor(np.load(path + '/random_1000_x_128.npy'))
    G_float32 = torch.tensor(G,dtype=torch .float32)
    data['G'].x = G_float32
    D = torch.tensor(np.load(path + '/random_207_x_128.npy'))
    D_float32 = torch.tensor(D,dtype=torch.float32)
    data['D'].x = D_float32

    #读取边
    #read D D edge matrix from csv
    DDS = pd.read_csv(path + '/dds_207.csv', index_col=0).to_numpy()
    DDS = torch.tensor(DDS.nonzero(),dtype=torch.long)
    data['D', 'DD', 'D'].edge_index = DDS
    
    #read G G edge matrix from csv
    GGS = pd.read_csv(path + '/ggs_1000.csv', index_col=0).to_numpy()
    GGS = torch.tensor(GGS.nonzero(),dtype=torch.long)
    data['G', 'GG', 'G'].edge_index = GGS
    
    #read G D edge matrix from csv
    # GDS = pd.read_csv(path + '/G_D_1000_207_alz.csv', index_col=0).to_numpy()
    # GDS = torch.tensor(GGS.nonzero(),dtype=torch.long)
    # data['G', 'GD', 'D'].edge_index = GDS


    GD = pd.read_csv(path + f'/X_{file_name}.txt',header=None)
    #replace 'G' with'' and 'D' with ''
    GD = GD.replace('G', '', regex=True).replace('D', '', regex=True).to_numpy().astype(int).T

    data['G', 'GD', 'D'].edge_index = torch.tensor(GD-1,dtype=torch.long)
     
    #this is y
    data.y = torch.tensor(pd.read_csv(path + f'/y_{file_name}.txt',header=None).to_numpy().reshape(-1),dtype=torch.float)
    
    # PyTorch tensor functionality:
    # data = data.pin_memory()
    # data = data.to('cuda:0', non_blocking=True)
    return data

train_data = get_train_data("train")
test_data = get_train_data("validation")
# test_data = get_train_data("test")

print("构建网络完毕——————————————————————————————————————————————————————————————————")

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

        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)
        self.DBN = DBN(visible_units=128, hidden_units=[128,128,128], k=5,use_gpu=False)
        # self.DBN = DBN(visible_units=128, hidden_units=[128,128,128], k=5,use_gpu=True)
        
        #  x_dict, edge_index_dict = 结点特征， 结点的边(GG，DD,GD也存在)
    def forward_GCN(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        #确定返回值
        return x_dict['G'],x_dict['D']

    def forward(self, x_dict, edge_index_dict,y=None,DBN = False):
        G,D = self.forward_GCN(x_dict, edge_index_dict)
        if DBN:
            G_num = G.shape[0]
            GD_node = torch.cat([G,D],dim=0)
            _,GD_node = self.DBN.train_static(GD_node,y)
            #split GD_node to G and D
            G,D = GD_node[:G_num],GD_node[G_num:]
        row, col = edge_index_dict[('G', 'GD', 'D')]  # train_data.edge_index_dict[('G', 'GD', 'D')].shape  =torch.Size([2, 9112])
        G = G[row]  # row = gene的id ，col = disease的id
        D = D[col]
        GD = torch.cat([G, D], dim=-1)  #  拼接G和D
        GD = self.lin1(GD).relu()
        GD = self.lin2(GD)
        return GD
        
    # 调用双线性decoder    
    def Builddecoder(self):
        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim, num_r=self.num_r, act=torch.sigmoid)(self.embeddings)


class FocalLoss(torch.nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

# HeteroGNN = HGCN模型
model = HeteroGNN(train_data.metadata(), hidden_channels=128, out_channels=96,num_layers=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data, model = train_data.to(device), model.to(device)   # cpu or GPU translate
test_data = test_data.to(device)

# with torch.no_grad():  # Initialize lazy modules.
#   out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
# ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)          # 1.指数衰减,lr=0.1
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.65)    # 2.固定步长衰减,lr=0.1
# torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[200, 300, 320, 340, 200], gamma=0.8) # 3.多步长衰减,lr=0.1
# CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)   # 4.余弦退火,lr=0.01. 

# 训练数据输入，训练模型
def train(DBN,model,data,label):
    model.train()
    optimizer.zero_grad()
    #x_dict:{节点类型：节点特征}
    #edge_index_dict:{(节点类型1，边类型，节点类型2)：边索引：[2,边数]}
    # if DBN，else GCN
    # if DBN:
    #     out = model(data.x_dict, data.edge_index_dict,y=label,DBN = DBN)
    #     #change out into long tensor
    #     label = torch.tensor(label,dtype=torch.long)
    #     loss_function = FocalLoss(gamma=1,reduction="none")
    #     #loss = F.cross_entropy(out, label)
    #     loss = loss_function(out, label)
    # else:
    out = model(data.x_dict, data.edge_index_dict,DBN = DBN)
    #change out into long tensor
    # out为预测值，label为真实值。
    label = torch.tensor(label,dtype=torch.long)
    loss_function = FocalLoss(gamma=1,reduction="none")
    #loss = F.cross_entropy(out, label)
    loss = loss_function(out, label)
        
    loss.backward()
    optimizer.step()
    return float(loss)

# 测试数据输入，测试模型
@torch.no_grad()
def test(model,data,label):
    model.eval()                  # ???????????
    pred = torch.softmax(model(data.x_dict, data.edge_index_dict,y=label),dim=-1)
    pred = torch.argmax(pred,dim=-1)   # argmax()函数功能：求取均值。 pred 为预测值
    acc = torch.mean(torch.tensor(1*(label==pred),dtype=torch.float))
    return acc, pred

def compute_aupr(y,yhat):
    pre,rec,_ = precision_recall_curve(y,yhat)
    return auc(rec,pre)

def f1_score(precison, recall):
    try:
        return 2*precison*recall /(precison + recall)
    except:
        return 0.0

print("开始训练——————————————————")
# 训练到一定阶段，接DBN
DBN_start = False
for epoch in range(1, 400):
    loss = train(DBN = DBN_start, model=model, data=train_data, label=train_data.y)
    test_acc, test_pred = test(model, test_data, test_data.y)
    # test_acc, test_pred = test(model, train_data, train_data.y)
    # if test_acc>0.8:
    #     DBN_start = True
    labels = test_data.y.cpu().numpy()
    test_pred = test_pred.cpu().numpy()
    AUC = roc_auc_score(labels,test_pred)            # AUROC
    aupr_score = compute_aupr(labels,test_pred)      # AUPRC
    avg_pre = average_precision_score(labels,test_pred)
    recal_score = recall_score(labels,test_pred)
    f1 = f1_score(avg_pre,recal_score)

    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test_acc: {test_acc:.4f}, AUC:{AUC:.4f}, f1:{f1:.4f}, Pre:{avg_pre:.4f}, Rec:{recal_score:.4f},AUPR:{aupr_score:.4f}')
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test_acc: {test_acc}, AUC:{AUC}, f1:{f1}, Pre:{avg_pre}, Rec:{recal_score},AUPR:{aupr_score}')


'''
train_data=
HeteroData(
  y=[9112],
  [1mG[0m={ x=[1000, 192] },
  [1mD[0m={ x=[217, 192] },
  [1m(D, DD, D)[0m={ edge_index=[2, 42858] },
  [1m(G, GG, G)[0m={ edge_index=[2, 872422] },
  [1m(G, GD, D)[0m={ edge_index=[2, 9112] }
  test_data = 
HeteroData(
  y=[2280],
  [1mG[0m={ x=[1000, 192] },
  [1mD[0m={ x=[217, 192] },
  [1m(D, DD, D)[0m={ edge_index=[2, 42858] },
  [1m(G, GG, G)[0m={ edge_index=[2, 872422] },
  [1m(G, GD, D)[0m={ edge_index=[2, 2280] }
)


train_data.edge_index_dict[('G', 'GD', 'D')].shape  =torch.Size([2, 9112])
test_data.edge_index_dict[('G', 'GD', 'D')].shape  =torch.Size([2, 2280])
test_data.x_dict['G'].shape = torch.Size([1000, 192])      # 结点特征向量
train_data.x_dict['G'].shape = torch.Size([1000, 192])
train_data.x_dict['D'].shape = torch.Size([217, 192])
train_data.y.shape = torch.Size([9112])
test_data.y.shape = torch.Size([2280])
test_data.edge_index_dict[('D', 'DD', 'D')].shape = torch.Size([2, 42858])

这里的GD网络使用的是训练集和测试集里面的边，不是GD矩阵。因此需要放入GD矩阵，预测和训练的时候使用train和test集合。

1.如何不加入结点特征，训练模型得到GD结果？

2.如何去掉GGS和DDS网络，只保留GDS网络来训练？

3.训练集和测试集的结点如何拿到属性特征？

'''