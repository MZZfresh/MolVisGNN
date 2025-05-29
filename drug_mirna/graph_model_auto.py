import torch.nn as nn
from torch_geometric.nn import GCNConv,HeteroConv,SAGEConv,GraphConv,GATConv,GraphNorm
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.utils import negative_sampling

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,roc_curve, auc,precision_recall_curve,average_precision_score,f1_score,recall_score
from config import device

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") 

class Conv1dNetwork(nn.Module):
    def __init__(self):
        super(Conv1dNetwork, self).__init__()

        self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1)

        self.conv2 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)

        self.conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)

        self.pool = nn.MaxPool1d(4)
        self.pool2 = nn.AdaptiveAvgPool1d(128)
        self.pool3 = nn.AdaptiveMaxPool1d(128)
        self.nor = nn.BatchNorm1d(128)
    def forward(self, x):

        x = x.unsqueeze(1)  

        x = F.relu(self.conv1(x))  
        x = self.pool(x)  

        x = F.relu(self.conv2(x))  
        x = self.pool(x)  

        x = F.relu(self.conv3(x))  
        x = self.pool(x)  

        x = x.view(x.size(0), -1) 
        x = x.unsqueeze(1) 
        x = self.pool2(x)
        x = x.squeeze(1)

        return x
    


class SliceAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super(SliceAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x: [B, C, D, H, W], D=8 is number of slices
        B, C, D, H, W = x.shape

        # Split into 4 parts along D (assume D=8)
        x1 = x[:, :, 0:2, :, :]
        x2 = x[:, :, 2:4, :, :]
        x3 = x[:, :, 4:6, :, :]
        x4 = x[:, :, 6:8, :, :]

        def process_part(part):
            B, C, D, H, W = part.shape
            out = part.view(B, C, D * H * W).transpose(1, 2)  # [B, N, C]
            out, _ = self.attn(out, out, out)
            return out.transpose(1, 2).view(B, C, D, H, W)

        x1 = process_part(x1)
        x2 = process_part(x2)
        x3 = process_part(x3)
        x4 = process_part(x4)

        # Concatenate along D (depth)
        return torch.cat([x1, x2, x3, x4], dim=2)  # [B, C, D=8, H, W]
    




class gate(nn.Module):
    def __init__(self):
        super(gate, self).__init__()
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def forward(self, x1,x2,x3):

        x1 = self.lin1(x1)
        x2 = self.lin1(x2)
        x3 = self.lin1(x3)
        x1 = self.lin2(x1)
        x2 = self.lin2(x2)
        x3 = self.lin2(x3)
        x1 = self.lin3(x1)
        x2 = self.lin3(x2)
        x3 = self.lin3(x3)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)
        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)
        x3 = x3.squeeze(1)
        x1 = self.sigmoid(x1)
        x2 = self.sigmoid(x2)
        x3 = self.sigmoid(x3)
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        if x3.dim() == 1:
            x3 = x3.unsqueeze(0)
        x = torch.cat((x1,x2,x3),dim=1)
        x = self.softmax(x)
        return x










class mlp_pre(torch.nn.Module):
    def __init__(self, num_in, num_hid1, num_hid2, num_out):
        super(mlp_pre, self).__init__()
        # 输入层与两层隐藏层基础
        self.l1 = torch.nn.Linear(num_in, num_hid1)
        self.bn1 = torch.nn.BatchNorm1d(num_hid1)
        self.l2 = torch.nn.Linear(num_hid1, num_hid2)
        self.bn2 = torch.nn.BatchNorm1d(num_hid2)
        # 增加两层额外隐藏层
        self.l3 = torch.nn.Linear(num_hid2, num_hid2)
        self.bn3 = torch.nn.BatchNorm1d(num_hid2)
        self.l4 = torch.nn.Linear(num_hid2, num_hid2)
        self.bn4 = torch.nn.BatchNorm1d(num_hid2)
        # 分类输出层
        self.classify = torch.nn.Linear(num_hid2, num_out)
        # 激活、正则化
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x2 = self.l4(x)
        x = self.l4(x)
        x = self.classify(x)
        return x,x2

class Directional3DProcessor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Directional3DProcessor, self).__init__()
        # 定义三个分组的卷积
        self.conv_fr = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_bb = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_tl = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, encoded_3d):  
        fr = encoded_3d[:, :, 0:2]  
        bb = encoded_3d[:, :, 2:4]
        tl = encoded_3d[:, :, 4:6]

        fr_out = self.conv_fr(fr)
        bb_out = self.conv_bb(bb)
        tl_out = self.conv_tl(tl)

        combined = torch.cat([fr_out, bb_out, tl_out], dim=2)  
        return combined  




class MolVisGNN(nn.Module):
    def __init__(self, ):
        super(MolVisGNN, self).__init__()
        self.conv1 = HeteroConv({
            ('miRNA', 'interacts', 'drug'): SAGEConv(256, 128),
            ('drug', 'interacts', 'miRNA'): SAGEConv(256, 128)
        },aggr='mean')
        self.conv2 = HeteroConv({
            ('miRNA', 'interacts', 'drug'): SAGEConv(128, 64),
            ('drug', 'interacts', 'miRNA'): SAGEConv(128, 64)
        },aggr='mean')
        self.conv3 = HeteroConv({
            ('miRNA', 'interacts', 'drug'): SAGEConv(64, 32),
            ('drug', 'interacts', 'miRNA'): SAGEConv(64, 32)
        },  aggr='mean')
        self.conv4 = HeteroConv({
            ('miRNA', 'interacts', 'drug'): SAGEConv(32, 16),
            ('drug', 'interacts', 'miRNA'): SAGEConv(32, 16)
        },  aggr='mean')


        self.mlp_pre = mlp_pre(64,32,16,1)
        self.lne = torch.nn.Linear(382, 256)
        self.nor = torch.nn.BatchNorm1d(128)
        self.nor2 = torch.nn.BatchNorm1d(16)
        self.nor3 = torch.nn.BatchNorm1d(128)
        self.resnet = Conv1dNetwork()
        self.relu = torch.nn.LeakyReLU(0.2)
        self.gate = gate()
        self.sp = Directional3DProcessor(128,64)


    def forward(self, x_dict, edge_index_dict,drug_2d_features, drug_3d_features):
        
        encode_3d = drug_3d_features
        encode_3d = self.sp(encode_3d)
        encode_3d = encode_3d.mean(dim=[2, 3, 4])
        encode_3d = F.adaptive_avg_pool1d(encode_3d, 256)

        drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 256)
        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'], 256)

        drug_1d_features = x_dict['drug']





        x = self.gate(x_dict['drug'],drug_2d_features, encode_3d)
        self.saved_x = x.detach().cpu().numpy()
        drug_2d_features = drug_2d_features * x[:,1].unsqueeze(1)
        drug_1d_features = drug_1d_features * x[:,0].unsqueeze(1)
        encode_3d = encode_3d * x[:,2].unsqueeze(1)
        x_dict['drug'] = torch.cat((drug_1d_features,drug_2d_features,encode_3d), dim=1)

        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'], 128)

        x_dict['drug'] = self.nor3(x_dict['drug'])
        drug_res = self.resnet(x_dict['drug'])
        x_dict['drug'] = drug_res + x_dict['drug']
        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'], 256)
        x_dict['miRNA'] = self.lne(x_dict['miRNA'])
        
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key:self.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key:self.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key:self.relu(x) for key, x in x_dict.items()}


        return x_dict
    





    def compute_loss(self, out, batch):


        edge_index = batch[('drug', 'interacts', 'miRNA')].edge_label_index
        labels = batch[('drug', 'interacts', 'miRNA')].edge_label


        drug_features = out['drug'][edge_index[0]]       # shape: [num_edges, drug_feature_dim]
        mirna_features = out['miRNA'][edge_index[1]]     # shape: [num_edges, mirna_feature_dim]


        edge_features = torch.cat([drug_features, mirna_features], dim=1)   # [num_edges, drug_dim + mirna_dim]


        scores,t = self.mlp_pre(edge_features)


        scores = scores.to(device).squeeze(1)
        labels = labels.to(device)


        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels.float())

        total_loss = loss

        return total_loss, scores, labels, edge_index,t

    
    def test(self, output, label):
        positive_class_probs = torch.sigmoid(output).detach().cpu().numpy()
        targets = label.cpu().numpy()


        auc = roc_auc_score(targets, positive_class_probs)
        aupr = average_precision_score(targets, positive_class_probs)

        # 将概率转换为二进制预测
        predicted = (positive_class_probs > 0.5).astype(int)

        # 计算其他指标
        accuracy = accuracy_score(targets, predicted)
        precision = precision_score(targets, predicted, zero_division=0)
        recall = recall_score(targets, predicted, zero_division=0)
        f1 = f1_score(targets, predicted, zero_division=0)

        return auc, aupr, accuracy, precision, recall, f1

