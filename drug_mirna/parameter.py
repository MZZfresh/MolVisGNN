import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import optuna
from utiles import *
from get_graph_auto import *
from tqdm import tqdm
from scaffold_split import get_graph_scaffold
from get_graph_auto import get_graph, load_mirna_features, lode1d_to_gpu, load_npy_to_gpu
# 以下两个函数原先在脚本中定义，这里内联实现
def get_drug_2d_features2(data, gpu_2d):
    drug_features_2d_tensors = []
    for idx in data:
        name = get_drug_name(str(int(idx)))
        feat = gpu_2d.get(name)
        drug_features_2d_tensors.append(feat)
    return torch.stack(drug_features_2d_tensors, dim=0)

def get_drug_features3d(node_idxs, drug_features_dict):
    drug_feature = []
    for node_idx in node_idxs:
        name = get_drug_name(str(int(node_idx)))
        drug_id = get_drug_id(name)
        fname = f"{drug_id}_output_encoded.npy"
        feat = drug_features_dict.get(fname, None)
        if feat is not None:
            drug_feature.append(feat)
    return torch.stack(drug_feature, dim=0)

from graph_model_auto import zhangzimai
from config import device

# —— 固定随机种子 ——
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# —— 训练+验证函数，返回验证集 AUC ——
def train_and_evaluate(lr, wd, args, train_data, val_data, gpu_2d, gpu_data_3d):
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

    # 完整训练
    for epoch in range(1, args.epochs + 1):
        model.train()
        d2 = get_drug_2d_features2(train_data['drug'].node_idx.tolist(), gpu_2d).to(device)
        d3 = get_drug_features3d(train_data['drug'].node_idx.tolist(), gpu_data_3d).to(device)
        out = model(train_data.x_dict, train_data.edge_index_dict, d2, d3)
        loss, _, _, _ ,_= model.compute_loss(out, train_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 最终在验证集计算 AUC
    model.eval()
    with torch.no_grad():
        d2v = get_drug_2d_features2(val_data['drug'].node_idx.tolist(), gpu_2d).to(device)
        d3v = get_drug_features3d(val_data['drug'].node_idx.tolist(), gpu_data_3d).to(device)
        out_val = model(val_data.x_dict, val_data.edge_index_dict, d2v, d3v)
        _, val_scores, val_labels, _ ,_= model.compute_loss(out_val, val_data)
        auc_val, _, _, _, _,_ = model.test(val_scores, val_labels)
    return auc_val

# —— Optuna 目标函数，最大化 AUC ——
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    wd = trial.suggest_loguniform('wd', 1e-6, 1e-3)
    auc = train_and_evaluate(lr, wd, args, train_data, val_data, gpu_2d, gpu_data_3d)
    return auc

# —— 主函数 ——
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',    type=int,   default=3000)
    parser.add_argument('--epochs',  type=int,   default=500)
    parser.add_argument('--sizes',   nargs='+',  type=int, default=[30,15,5,1])
    parser.add_argument('--conv_type',     type=str, choices=['local','global','full'], default='full')
    parser.add_argument('--in_channels',   type=int, default=128)
    parser.add_argument('--out_channels',  type=int, default=32)
    parser.add_argument('--out_channels_gat', type=int, default=32)
    parser.add_argument('--hidden_dim',    type=int, default=128)
    parser.add_argument('--hidden_dim2',   type=int, default=128)
    parser.add_argument('--global_dim',    type=int, default=128)
    parser.add_argument('--num_layers',    type=int, default=1)
    parser.add_argument('--num_heads',     type=int, default=4)
    parser.add_argument('--ff_dropout',    type=float, default=0.2)
    parser.add_argument('--attn_dropout',  type=float, default=0.2)
    parser.add_argument('--skip',          type=int, default=10)
    parser.add_argument('--dist_count_norm', type=int, default=1)
    parser.add_argument('--num_centroids',   type=int, default=10)
    parser.add_argument('--no_bn', action='store_true')
    parser.add_argument('--norm_type',      type=str, choices=['layer_norm','batch_norm'], default='batch_norm')
    args = parser.parse_args()

    # 数据准备
    set_random_seed(args.seed)
    mirna_1d = load_mirna_features('graph/data/kmer_features.csv', device=device)
    gpu_1d = lode1d_to_gpu('graph/data/drug_1d_features.csv', device=device)
    gpu_2d = lode1d_to_gpu('graph/data/128_2d.csv', device=device)
    npy_dir = 'graph/data/ddi_encodedre1'
    gpu_data_3d = load_npy_to_gpu([os.path.join(npy_dir, f) for f in os.listdir(npy_dir)], device)
    train_data, val_data, test_data = get_graph_scaffold(gpu_1d, mirna_1d)

    # 启动 Optuna，最大化 AUC
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print('Best hyperparameters:', study.best_params)
    print('Best validation AUC:', study.best_value)
    study.trials_dataframe().to_csv('optuna_auc_results.csv', index=False)
