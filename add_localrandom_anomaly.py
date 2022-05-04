import numpy as np
import pandas as pd
import networkx as nx

dataset = 'cora'
ratio = 0.1

edges = pd.read_csv('data/'+dataset+'/'+dataset+'.csv').values
# edges = pd.read_csv('data/BlogCatalog/BlogCatalog.edge',header=None,sep='\t').values
# edges=pd.read_csv('data/citeseer/citeseer.edge', header=None, sep='\t').values
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

degree_table = list(g.degree)
degree_table.sort(key=lambda x:x[1], reverse=True)
source_nodes = degree_table[:int(len(degree_table)*0.2)]    # top 20% of the nodes with max degree are selected
print(source_nodes)
source_nodes = [x[0] for x in source_nodes]
print(source_nodes)

total_thre = int(ori_edges_num*ratio)
while(total_thre):
    nodepair = np.random.choice(source_nodes, 2, replace=False)     # 不能选重复元素
    while(g.has_edge(nodepair[0],nodepair[1])):
        nodepair = np.random.choice(source_nodes, 2, replace=False)
    g.add_edge(nodepair[0], nodepair[1], weight=1)
    total_thre = total_thre - 1

print(g.number_of_edges())
print((g.number_of_edges()-ori_edges_num)/ori_edges_num)

new_edge_table = nx.to_pandas_edgelist(g)
new_edge_table.to_csv('data/'+dataset+'/localr_'+dataset+str(ratio)+'.csv', index=False)