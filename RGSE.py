import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from model import *
import torch
import torch.nn as nn
import torch.optim as optim


def read_graph(edges, nodes):
    g = nx.Graph()
    for i in range(nodes):
        g.add_node(i)
    for edge in edges:
        origin_city = int(edge[0])
        dest_city = int(edge[1])
        if not g.has_edge(origin_city, dest_city):
            g.add_edge(origin_city, dest_city, weight=0)
        g.get_edge_data(origin_city, dest_city)['weight'] += 1
    return g


def extract_features(graph, x, y):
    common_neighbors = list(nx.common_neighbors(graph, x, y))
    x_degree = y_degree = 0
    for i in list(nx.neighbors(graph, x)):
        x_degree += graph.get_edge_data(x, i)['weight']
    for i in list(nx.neighbors(graph, y)):
        y_degree += graph.get_edge_data(y, i)['weight']

    z_weight_sum = 0
    xz = zy = 0
    for z in common_neighbors:
        sz = 0
        for i in list(nx.neighbors(graph, z)):
            sz += graph.get_edge_data(i, z)['weight']
        z_weight_sum += sz

        xz += graph.get_edge_data(x, z)['weight']
        zy += graph.get_edge_data(z, y)['weight']
    if len(common_neighbors) == 0:
        feat1 = 0
    else:
        feat1 = xz / z_weight_sum
    if x_degree > 0:
        feat2 = xz / x_degree
    else:feat2 = 0
    if y_degree > 0:
        feat3 = zy / y_degree
    else:feat3 = 0
    features = [feat1, feat2, feat3]
    return features


def get_needed_data(g):
    newdata = []
    node_nums = g.number_of_nodes()
    for e in g.edges:
        newdata.append([e[0], e[1], 1])    # pos
        for _ in range(1):
            j = np.random.randint(node_nums)    # left neg
            while g.has_edge(e[0], j):
                j = np.random.randint(node_nums)
            newdata.append([e[0], j, 0])
        for _ in range(1):
            j = np.random.randint(node_nums)    # right neg
            while g.has_edge(e[1], j):
                j = np.random.randint(node_nums)
            newdata.append([e[1], j, 0])
    return np.array(newdata)


def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


dataset = 'cora'
ratio = 0.1
edges = pd.read_csv('data/' + dataset + '/ano_'+dataset+str(ratio)+'.csv').values

nodes_nums = edges.max() + 1
g = read_graph(edges, nodes_nums)

adj = np.array(nx.adjacency_matrix(g).todense())
adj = torch.from_numpy(adj).float().cuda()

edges = get_needed_data(g)
edges_left = torch.from_numpy(edges[:, 0]).long().cuda()
edges_right = torch.from_numpy(edges[:, 1]).long().cuda()
res_label = edges[:, -1]
edges_features = np.zeros((len(edges), 3))

# you can save cn features and just load them
print('extract features')
for i, edge in enumerate(edges):
    features = extract_features(g, int(edge[0]), int(edge[1]))
    edges_features[i] = features
print('extract features finished!')

edges = pd.read_csv('data/' + dataset + '/ano_'+dataset+str(ratio)+'.csv').values
ano_edges_left = torch.from_numpy(edges[:, 0]).long().cuda()
ano_edges_right = torch.from_numpy(edges[:, 1]).long().cuda()
ano_label = edges[:, -1]
print('anomaly rate:', len(ano_label[ano_label>0])/len(ano_label[ano_label==0]))

ori_edges_features = np.zeros((len(edges), 3))
for i, edge in enumerate(edges):
    features = extract_features(g, int(edge[0]), int(edge[1]))
    ori_edges_features[i] = features

model = MyModel_combine(32, 16, 16, 3, 32, 16, dataset, ratio)
model = model.cuda()

all_data = torch.from_numpy(edges_features).float().cuda()
res_label = torch.from_numpy(res_label).long().cuda()
ori_data = torch.from_numpy(ori_edges_features).float().cuda()

criterion = nn.CrossEntropyLoss(torch.Tensor([1, 2]).cuda())
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

weight_tensor, norm = compute_loss_para(adj)
epochs = 300
best_auc = 0
for epoch in range(epochs):
    model.train()
    out, _ = model(edges_left, edges_right, all_data)
    loss = criterion(out, res_label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch '+str(epoch)+' loss = ' + str(loss.item()))

    model.eval()
    pre, emb = model(ano_edges_left, ano_edges_right, ori_data)
    pre = pre[:, 0]
    auc = roc_auc_score(ano_label, pre.cpu().detach().numpy())
    if auc > best_auc:
        best_auc = auc
    print('auc = '+str(auc))
print(best_auc)
