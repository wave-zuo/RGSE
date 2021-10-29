import numpy as np
import pandas as pd
import networkx as nx
from model import *
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    print('pos weight=',pos_weight)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm

# contrastive loss is used by Graph-mlp's implementation
def Ncontrast(x_dis, adj_label, tau = 1):
    x_dis = torch.exp(x_dis/tau)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))).mean()
    return loss

def get_feature_dis(x):
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

dataset = 'cora'
ratio = 0.1
edges = pd.read_csv('data/' + dataset + '/ano_'+dataset+str(ratio)+'.csv').values

node_nums = edges.max()+1
g = nx.Graph()
g.add_nodes_from(range(node_nums))

labels = edges[:, -1]
for e in edges:
    g.add_edge(e[0], e[1])

adj = np.array(nx.adjacency_matrix(g).todense())
adj = torch.from_numpy(adj).float().cuda()
weight_tensor, norm = compute_loss_para(adj)

model = GAElinear(node_nums, 128, 32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model = model.cuda()
epochs = 50
best_auc = 0
for epoch in range(epochs):
    model.train()
    recovered, z = model(adj)
    x_dis = get_feature_dis(z)
    loss_Ncontrast = Ncontrast(x_dis, adj, tau=1)   # use tau=0.1 in Enron

    loss = norm * F.binary_cross_entropy(recovered.view(-1), adj.view(-1), weight=weight_tensor)
    total_loss = loss + 1*loss_Ncontrast
    cur_loss = total_loss.item()
    print('epoch: '+str(epoch)+', loss = '+str(cur_loss))
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # save embeddings
    # np.save('data/' + dataset + '/ae_embed_nloss2' + str(ratio) + '.npy', z.cpu().detach().numpy())

    model.eval()
    res, z = model(adj)

    res = (res + res.t())/2
    res = (res - adj)**2
    res = res.cpu().detach().numpy()

    pre = []
    for e in edges:
        pre.append(res[e[0], e[1]])
    pre = np.array(pre)
    auc = roc_auc_score(labels, pre)
    print('auc = ', auc)

