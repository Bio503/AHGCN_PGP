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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
# from sklearn.metrics import accuracy_score   #计算accuracy值
# from sklearn.metrics import roc_curve     # 画图用
from sklearn.metrics import recall_score
# from sklearn.metrics import confusion_matrix

#path = osp.join(osp.dirname(osp.realpath(__file__)), '..\..\data\DBLP')
## We initialize conference node features with a single one-vector as feature:
#dataset = DBLP(path, transform=T.Constant())
#data = dataset[0]
#print(data)
def get_data(file_name=None):
    # path = 'bio'
    path = "data_evalution"
    # path = "ad_data"

    data = HeteroData()
    #读取节点特征
    G = torch.as_tensor(np.load(path + '/g_gd_ppi_feature_node2vec_1000.npy'))
    # G = torch.tensor(np.load(path + '/g_feature_nl.npy'))
    data['G'].x = G
    D = torch.as_tensor(np.load(path + '/d_gd_hpo_feature_node2vec_1000.npy'))
    # D = torch.tensor(np.load(path + '/d_feature_nl.npy'))
    data['D'].x = D
    #读取边index
    #read D D edge matrix from csv
    DDS = pd.read_csv(path + '/dds_207.csv', index_col=0).to_numpy()
    DDS = torch.as_tensor(DDS.nonzero(),dtype=torch.long)
    data['D', 'DD', 'D'].edge_index = DDS
    
    #read G G edge matrix from csv
    GGS = pd.read_csv(path + '/ggs_1000.csv', index_col=0).to_numpy()
    GGS = torch.as_tensor(GGS.nonzero(),dtype=torch.long)
    data['G', 'GG', 'G'].edge_index = GGS
    
    #read G D edge matrix from csv
    GD = pd.read_csv(path + f'/X_{file_name}.txt',header=None)
    #replace 'G' with'' and 'D' with ''
    GD = GD.replace('G', '', regex=True).replace('D', '', regex=True).to_numpy().astype(int).T
    data['G', 'GD', 'D'].edge_index = torch.as_tensor(GD-1,dtype=torch.long)
    
    #read G D edge matrix from csv
    #this is y
    data.y = torch.as_tensor(pd.read_csv(path + f'/y_{file_name}.txt',header=None).to_numpy().reshape(-1),dtype=torch.float)

    # PyTorch tensor functionality:
    # data = data.pin_memory()
    # data = data.to('cuda:0', non_blocking=True)
    return data
def get_edge_feature(data):
    edge_indexs = data['G', 'GD', 'D']['edge_index'].T
    edges = []
    for edge_index in edge_indexs:
        G = data['G'].x[edge_index[0]]
        D = data['D'].x[edge_index[1]]
        GD = torch.cat([G,D])
        edges.append(GD[None,:])
    edges = torch.cat(edges)
    return edges

data_train = get_data("train")
data_test = get_data("validation")
edges_train = get_edge_feature(data_train)
edges_test = get_edge_feature(data_test)
label_train = torch.as_tensor(data_train.y,dtype=torch.long)
label_test = torch.as_tensor(data_test.y,dtype=torch.long)

model = torch.nn.Linear(edges_train.shape[-1],2) # 输入特征，输出特征

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edges_train, model, edges_test,label_train = edges_train.to(device), model.to(device), edges_test.to(device),label_train.to(device)
data_test = data_test.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train(model,data,label):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out, label)
    loss.backward()
    optimizer.step()
    return float(loss)

def compute_aupr(y,yhat):
    pre,rec,_ = precision_recall_curve(y,yhat)
    return auc(rec,pre)

def f1_score(precison, recall):
    try:
        return 2*precison*recall /(precison + recall)
    except:
        return 0.0


@torch.no_grad()
def test(model,data,label):
    model.eval()
    pred = torch.softmax(model(data),dim=-1)
    pred = torch.argmax(pred,dim=-1)   # argmax()函数功能：求取均值 
    acc = torch.mean(torch.as_tensor(1*(label==pred),dtype=torch.float))
    return acc,pred
for epoch in range(1, 500):
    loss = train(model,edges_train,label_train)
    test_acc,test_pred = test(model,edges_test,label_test)

    labels = data_test.y.cpu().numpy()
    test_pred = test_pred.cpu().numpy()
    # accuracy = accuracy_score(labels,test_pred)
    AUC = roc_auc_score(labels,test_pred)
    aupr_score = compute_aupr(labels,test_pred)
    avg_pre = average_precision_score(labels,test_pred)
    recal_score = recall_score(labels,test_pred)
    f1 = f1_score(avg_pre,recal_score)

    # print(f'Epoch: {epoch:03d}, Loss: {loss:.7f}, T_acc: {test_acc:.8f}, AUC:{AUC:.8f}, accuracy: {accuracy:.8f}')
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, T_acc: {test_acc}, AUC:{AUC}, f1:{f1:.6f}, Pre:{avg_pre:.6f}, Rec:{recal_score:.6f},AUPR:{aupr_score:.6f}')


pass

# bio中的原始数据：Epoch: 999, Loss: 0.4593, Test: 0.7746
# 换成ad的1000个小样本性能完全为无。Epoch: 1999, Loss: 0.3526, Test: 0.3240
