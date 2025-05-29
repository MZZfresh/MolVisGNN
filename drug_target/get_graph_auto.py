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
from utiles_auto import *
from config import device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



def get_drug_data(gpu_1d):
    df = pd.read_csv('drug_index.csv')
    
    drug_1d_feature = []
    for drug_name in df.iloc[:, 0]:
        drug_feature = gpu_1d[drug_name].to(device)
        drug_1d_feature.append(drug_feature)
    # print(drug_1d_feature)
    # drug_1d_feature = torch.tensor(drug_1d_feature)
    return drug_1d_feature

def get_Protein_data(Protein_1d_feature):
    df = pd.read_csv('protein_index.csv')
    Protein_feature_1d = []
    for Protein_name in df.iloc[:, 0]:
        # Protein_id = get_Protein_id(Protein_name)
        Protein_feature = Protein_1d_feature[Protein_name].to(device)
        Protein_feature_1d.append(Protein_feature)
    # Protein_feature_1d = torch.tensor(Protein_feature_1d)
    return Protein_feature_1d








def get_graph(gpu_1d, Protein_1d_feature):

    drug_feature = get_drug_data(gpu_1d)

    Protein_feature = get_Protein_data(Protein_1d_feature)

    drug_feature = torch.stack(drug_feature).to(device)
    Protein_feature = torch.stack(Protein_feature).to(device)


    
    edge = pd.read_csv('relation.csv')
    

    graph = HeteroData()
    

    graph['Protein'].x = Protein_feature.to(torch.float32)
    

    graph['drug'].x = drug_feature.to(torch.float32)


    Protein_idx = torch.arange(Protein_feature.size(0), dtype=torch.float32)
    drug_idx = torch.arange(drug_feature.size(0), dtype=torch.float32)
    graph['Protein'].node_idx = Protein_idx
    graph['drug'].node_idx = drug_idx




    edge_index = torch.tensor(edge.values, dtype=torch.long).t().contiguous().to(device)


    reversed_edge_index = edge_index.flip(0)

    num_drug_nodes = graph['drug'].x.size(0)
    num_Protein_nodes = graph['Protein'].x.size(0)
    # num_nodes = num_drug_nodes + num_Protein_nodes

    graph['Protein'].num_nodes = num_Protein_nodes
    graph['drug'].num_nodes = num_drug_nodes

    graph['drug', 'interacts', 'Protein'].edge_index = edge_index
    graph['Protein', 'interacts', 'drug'].edge_index = reversed_edge_index
    

    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
        # add_negative_train_samples=True,
        # add_negative_val_samples=True,
        # add_negative_test_samples=True,
        edge_types=('drug', 'interacts', 'Protein'),
        rev_edge_types=('Protein', 'interacts', 'drug')
    )
    train_data, val_data, test_data = transform(graph)

    return train_data, val_data, test_data


