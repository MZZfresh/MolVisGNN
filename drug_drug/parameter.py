import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import optuna
from tqdm import tqdm

from get_graph import get_graph2, load_npy_to_gpu, lode1d_to_gpu, lode2d_to_gpu, get_drug_features
from graph_model import zhangzimai, Autoencoder
from graph_model import *
from torch_geometric.loader import NeighborSampler, GraphSAINTRandomWalkSampler
from config import device

# Inline missing 3d loaders
def get_drug_features3d(node_idxs, idx_to_node, drug_features_dict):
    drug_feature = []
    for node_idx in node_idxs:
        name = idx_to_node.get(node_idx.item())
        if not name:
            continue
        fname = f"{name}_output_encoded.npy"
        feat = drug_features_dict.get(fname)
        if feat is not None:
            drug_feature.append(feat)
    if not drug_feature:
        raise ValueError("No valid 3D features found.")
    return torch.stack(drug_feature, dim=0).to(device)
def get_drug_features(node_idxs, idx_to_node, drug_features_dict):
    drug_feature = []
    # 根据节点索引查找药物名称
    for node_idx in node_idxs:
        drug_name = idx_to_node.get(node_idx.item(), None)
        # print(drug_name)
        # 根据药物名称查找特征
        drug_features = drug_features_dict.get(drug_name, None)
        if drug_features is not None:
            # print(f"Found features for {drug_name}: {drug_features}")
            # 如果 drug_features 不是 None，则添加到列表中
            drug_feature.append(drug_features)
        else:
            print(f"No features found for {drug_name}. Skipping.")
            # 如果没有特征，跳过该药物
            continue

    drug_feature = torch.stack(drug_feature, dim=0)
    return drug_feature
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# train & evaluate returning val loss
def train_and_evaluate(lr, wd, args, train_data, val_data, idx_to_node, gpu_1d, gpu_2d, gpu_data_3d):
    set_random_seed(args.seed)
    model = zhangzimai(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_dim,
        hidden_channels2=args.hidden_dim2,
        out_channels_gat=args.out_channels_gat,
        out_channels=args.out_channels,
        global_dim=args.global_dim,
        num_layers=args.num_layers,
        heads=args.num_heads,
        ff_dropout=args.ff_dropout,
        attn_dropout=args.attn_dropout,
        spatial_size=len(args.sizes),
        skip=args.skip,
        dist_count_norm=args.dist_count_norm,
        conv_type=args.conv_type,
        num_centroids=args.num_centroids,
        no_bn=args.no_bn,
        norm_type=args.norm_type
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # full train
    for epoch in range(1, args.epochs+1):
        model.train()
        d1 = get_drug_features(train_data.node_idx, idx_to_node, gpu_1d).to(device)
        d2 = get_drug_features(train_data.node_idx, idx_to_node, gpu_2d).to(device)
        d3 = get_drug_features3d(train_data.node_idx, idx_to_node, gpu_data_3d)
        out ,weight= model(train_data, d1, d2, d3)
        loss, _, _, _ ,_= model.compute_loss(out, train_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # val evaluate
    with torch.no_grad():
        d1v = get_drug_features(val_data.node_idx, idx_to_node, gpu_1d).to(device)
        d2v = get_drug_features(val_data.node_idx, idx_to_node, gpu_2d).to(device)
        d3v = get_drug_features3d(val_data.node_idx, idx_to_node, gpu_data_3d)
        out_val,weight = model(val_data, d1v, d2v, d3v)
        _, val_scores, val_labels, _ , _ = model.compute_loss(out_val, val_data)
        auc_val,aupr_val,accuracy_val,precision_val,recall = model.test(val_scores, val_labels)
        print(auc_val, aupr_val,accuracy_val,precision_val,recall)
    return accuracy_val  # 返回 AUC 而不是 loss

# Optuna objective
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
    return train_and_evaluate(
        lr, wd, args,
        train_data, val_data, idx_to_node,
        gpu_1d, gpu_2d, gpu_data_3d
    )

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--sizes', nargs='+', type=int, default=[30,15,5,1])
    parser.add_argument('--conv_type', choices=['local','global','full'], default='full')
    parser.add_argument('--in_channels', type=int, default=256)
    parser.add_argument('--out_channels', type=int, default=32)
    parser.add_argument('--out_channels_gat', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--hidden_dim2', type=int, default=128)
    parser.add_argument('--global_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--ff_dropout', type=float, default=0.2)
    parser.add_argument('--attn_dropout', type=float, default=0.2)
    parser.add_argument('--skip', type=int, default=10)
    parser.add_argument('--dist_count_norm', type=int, default=1)
    parser.add_argument('--num_centroids', type=int, default=10)
    parser.add_argument('--no_bn', action='store_true')
    parser.add_argument('--norm_type', choices=['layer_norm','batch_norm'], default='batch_norm')
    args = parser.parse_args()

    set_random_seed(args.seed)
    # load features
    npy_dir = 'graph_DDI/data/ddi_encodedre1'
    gpu_data_3d = load_npy_to_gpu([os.path.join(npy_dir,f) for f in os.listdir(npy_dir)], device)
    gpu_1d = lode1d_to_gpu('graph_DDI/data/drug_1d_fingerprints.csv', device=device)
    gpu_2d = lode2d_to_gpu('graph_DDI/data/drug_2d.csv', device=device)
    train_data, val_data, test_data, idx_to_node = get_graph2()

    # Optuna search，改为最大化 AUC
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('Best params (by AUC):', study.best_params)
    print('Best val AUC:', study.best_value)
    study.trials_dataframe().to_csv('optuna_auc_results.csv', index=False)
