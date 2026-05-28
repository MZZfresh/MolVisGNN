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

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# gpu_1d = lode1d_to_gpu('graph/data/128_2d.csv',device=device)
# mirna_1d_feature = load_mirna_features('graph/data/kmer_features.csv',device=device)

def get_drug_data(gpu_1d):
    df = pd.read_csv('drug_mapping.csv')
    drug_1d_feature = []
    for drug_name in df.iloc[:, 0]:
        drug_feature = gpu_1d[drug_name].to(device)
        drug_1d_feature.append(drug_feature)

    # drug_1d_feature = torch.tensor(drug_1d_feature)
    return drug_1d_feature

def get_mirna_data(mirna_1d_feature):
    df = pd.read_csv('ncRNA_mapping.csv')
    mirna_feature_1d = []
    for mirna_name in df.iloc[:, 0]:
        mirna_id = get_mirna_id(mirna_name)
        mirna_feature = mirna_1d_feature[mirna_id].to(device)
        mirna_feature_1d.append(mirna_feature)
    # mirna_feature_1d = torch.tensor(mirna_feature_1d)
    return mirna_feature_1d



def get_graph_global_link_pred(gpu_1d, mirna_1d_feature):

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    drug_feature = torch.stack(get_drug_data(gpu_1d)).to(device)
    mirna_feature = torch.stack(get_mirna_data(mirna_1d_feature)).to(device)
    drug_feature = F.adaptive_avg_pool1d(drug_feature.unsqueeze(0), 382).squeeze(0)


    edge_df = pd.read_csv('graph/data/edge_index.csv')
    edge_index = torch.tensor(edge_df.values, dtype=torch.long).t().contiguous().to(device)
    num_edges = edge_index.size(1)


    perm = torch.randperm(num_edges, device=device)
    train_size = int(num_edges * 0.7)
    val_size = int(num_edges * 0.1)

    train_pos_edge = edge_index[:, perm[:train_size]]
    val_pos_edge = edge_index[:, perm[train_size:train_size+val_size]]
    test_pos_edge = edge_index[:, perm[train_size+val_size:]]

    num_miRNA = mirna_feature.size(0)
    num_drug = drug_feature.size(0)


    all_pos_edges = set((int(u), int(v)) for u, v in edge_index.t())


    used_neg_edges = set()


    def make_graph(pos_edge_index, split_name):
        g = build_subgraph(mirna_feature, drug_feature, pos_edge_index, num_miRNA, num_drug)
        num_pos = pos_edge_index.size(1)

        current_pos = set((int(u), int(v)) for u, v in pos_edge_index.t())


        neg_edges = []
        while len(neg_edges) < num_pos:
            m = np.random.randint(0, num_miRNA)
            d = np.random.randint(0, num_drug)

            if (
                (m, d) not in all_pos_edges    
                and (m, d) not in current_pos  
                and (m, d) not in used_neg_edges  
            ):
                neg_edges.append((m, d))
                used_neg_edges.add((m, d))

        neg_edge_index = torch.tensor(neg_edges, dtype=torch.long).t().contiguous().to(device)


        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_label = torch.cat([
            torch.ones(num_pos, device=device),
            torch.zeros(num_pos, device=device)
        ]).float()


        g['miRNA', 'interacts', 'drug'].edge_label_index = edge_label_index
        g['miRNA', 'interacts', 'drug'].edge_label = edge_label

        g['drug', 'interacts', 'miRNA'].edge_label_index = edge_label_index.flip(0)
        g['drug', 'interacts', 'miRNA'].edge_label = edge_label

 
        g['miRNA'].node_idx = torch.arange(num_miRNA, device=device)
        g['drug'].node_idx = torch.arange(num_drug, device=device)


        iso = check_isolated_nodes(g)
        print(f"[{split_name}] 正样本：{num_pos} | 负样本：{num_pos} | 孤立节点：{iso}")

        return g

    # ========== 6. 依次构建数据集（负边绝不重复） ==========
    train_data = make_graph(train_pos_edge, "Train")
    val_data = make_graph(val_pos_edge, "Val")
    test_data = make_graph(test_pos_edge, "Test")

    return train_data, val_data, test_data




def get_graph(gpu_1d, mirna_1d_feature):

    drug_feature = get_drug_data(gpu_1d)

    mirna_feature = get_mirna_data(mirna_1d_feature)
    drug_feature = torch.stack(drug_feature).to(device)
    mirna_feature = torch.stack(mirna_feature).to(device)


    drug_feature = F.adaptive_avg_pool1d(drug_feature.unsqueeze(0), 382).squeeze(0)
    
    edge = pd.read_csv('edge_index.csv')
    

    graph = HeteroData()
    

    graph['miRNA'].x = mirna_feature.to(torch.float32)
    

    graph['drug'].x = drug_feature.to(torch.float32)

    mirna_idx = torch.arange(mirna_feature.size(0), dtype=torch.float32)
    drug_idx = torch.arange(drug_feature.size(0), dtype=torch.float32)
    graph['miRNA'].node_idx = mirna_idx
    graph['drug'].node_idx = drug_idx




    edge_index = torch.tensor(edge.values, dtype=torch.long).t().contiguous().to(device)

    reversed_edge_index = edge_index.flip(0)

    num_drug_nodes = graph['drug'].x.size(0)
    num_mirna_nodes = graph['miRNA'].x.size(0)
    # num_nodes = num_drug_nodes + num_mirna_nodes

    graph['miRNA'].num_nodes = num_mirna_nodes
    graph['drug'].num_nodes = num_drug_nodes

    # 设置“interaction”边
    graph['miRNA', 'interacts', 'drug'].edge_index = edge_index
    graph['drug', 'interacts', 'miRNA'].edge_index = reversed_edge_index

    transform = RandomLinkSplit(
        num_val=0.2,
        num_test=0.1,
        disjoint_train_ratio=0,
        neg_sampling_ratio=1.0,
        # add_negative_train_samples=True,
        # add_negative_val_samples=True,
        # add_negative_test_samples=True,
        edge_types=('drug', 'interacts', 'miRNA'),
        rev_edge_types=('miRNA', 'interacts', 'drug')
    )
    
    train_data, val_data, test_data = transform(graph)

    return train_data, val_data, test_data




