import torch_geometric
import torch
import torch.nn.functional as F
from torch_geometric.data import Data,HeteroData
import torch_geometric.utils as utils
from torch_geometric.transforms import RandomLinkSplit
from torch_cluster import random_walk
from torch_geometric.utils import train_test_split_edges
from sklearn.model_selection import train_test_split
import pandas as pd
from utiles import *





def get_graph():
    edge = pd.read_csv('ddis.csv')
    edges = edge[['d1', 'd2']].values
    all_nodes = list(set([drug for edge_pair in edges for drug in edge_pair]))
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    edge_index = torch.tensor([[node_to_idx[d1], node_to_idx[d2]] for d1, d2 in edges], dtype=torch.long).t().contiguous()
    
    graph = Data(edge_index=edge_index)
    graph.node_idx = torch.tensor([node_to_idx[node] for node in all_nodes], dtype=torch.long)
    transform = RandomLinkSplit(num_val=0.05,num_test=0.1,is_undirected=False,disjoint_train_ratio=0)
    train_data, val_data, test_data = transform(graph)
    mapping_df = pd.DataFrame(list(idx_to_node.items()), columns=['index', 'drug_name'])
    mapping_df.to_csv('index_to_drug_name.csv', index=False)
    return train_data, val_data, test_data, idx_to_node



