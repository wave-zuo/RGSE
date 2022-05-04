import numpy as np
import pandas as pd
import networkx as nx

dataset = 'cora'
edges = pd.read_csv('data/'+dataset+'/'+dataset+'.csv').values
# edges = pd.read_csv('data/BlogCatalog/BlogCatalog.edge',header=None,sep='\t').values
edge_nums = len(edges)
node_nums = edges.max()+1
g = nx.Graph()
g.add_nodes_from(range(node_nums))
for e in edges:
    if not g.has_edge(e[0], e[1]):
        g.add_edge(e[0], e[1], weight=0)
print(g.number_of_nodes())
ori_edges_num = g.number_of_edges()
print(ori_edges_num)

total_anomaly = 0
# random select M source nodes
M = int(node_nums*0.4)
# in our experiments, we fix T=20
T = 20
ratio = 0.1
total_thre = ori_edges_num*ratio
source_nodes = np.random.randint(0, node_nums, M)
for u in source_nodes:
    neighs = list(g.adj[u])
    neigh_nums = len(neighs)
    thre = ratio*neigh_nums
    candidate_nodes = np.random.randint(0, node_nums, int(T*thre))
    dist = []
    for c in candidate_nodes:
        if nx.has_path(g, u, c):
            dis = nx.shortest_path_length(g, source=u, target=c)
            dist.append([c, dis])   # [node,dist]
        else:
            dist.append([c, 100000])
    dist.sort(key=lambda x: x[1], reverse=True)
    cnt = 0
    for v in dist:
        if (u != v[0]) and (v[0] not in neighs) and (not g.has_edge(u, v[0])):
            g.add_edge(u, v[0], weight=1)
            cnt = cnt + 1
            total_anomaly = total_anomaly + 1
        if cnt >= thre:
            break
        if total_anomaly >= total_thre:
            break
    if total_anomaly >= total_thre:
        break
print('new graph edges = '+str(g.number_of_edges()))
print('anomaly ratio='+str((g.number_of_edges()-ori_edges_num)/ori_edges_num))

new_edge_table = nx.to_pandas_edgelist(g)
new_edge_table.to_csv('data/'+dataset+'/ano_'+dataset+str(ratio)+'.csv', index=False)