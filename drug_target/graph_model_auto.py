import torch.nn as nn
from torch_geometric.nn import GCNConv,HeteroConv,SAGEConv,GraphConv,GATConv
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,roc_curve, auc,precision_recall_curve,average_precision_score,f1_score,recall_score
import math
from config import device
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

class Conv1dNetwork(nn.Module):
    def __init__(self):
        super(Conv1dNetwork, self).__init__()

        self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1)

        self.conv2 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)

        self.conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)

        self.pool = nn.MaxPool1d(4)
        self.pool2 = nn.AdaptiveAvgPool1d(128)
        self.pool3 = nn.AdaptiveMaxPool1d(512)
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
        x = self.nor(x)
        return x
    

class SliceAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super(SliceAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x: [B, C, D, H, W], D=6 is number of slices
        B, C, D, H, W = x.shape

        # Split into 3 parts along D (assume D=6)
        x1 = x[:, :, 0:2, :, :]
        x2 = x[:, :, 2:4, :, :]
        x3 = x[:, :, 4:6, :, :]

        def process_part(part):
            B, C, D, H, W = part.shape
            out = part.view(B, C, D * H * W).transpose(1, 2)  # [B, N, C]
            out, _ = self.attn(out, out, out)
            return out.transpose(1, 2).view(B, C, D, H, W)

        x1 = process_part(x1)
        x2 = process_part(x2)
        x3 = process_part(x3)

        # Concatenate along D (depth)
        return torch.cat([x1, x2, x3], dim=2)  # [B, C, D=6, H, W]
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 16, 4, 256, 256)
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 32, 2, 128, 128)
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 64, 1, 64, 64)
            nn.ReLU(True)
        )

        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 128, 1, 32, 32)
            nn.ReLU(True)
        )

        # Attention block after bottleneck
        self.attn_block = SliceAttentionBlock(embed_dim=128, num_heads=4)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 4, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck_conv(x)
        x = self.attn_block(x)  # 融合 attention 后的编码表示
        x_recon = self.decoder(x)
        return x, x_recon


class Autoencoder2(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 16, 4, 256, 256)
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 32, 2, 128, 128)
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 64, 1, 64, 64)
            nn.ReLU(True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 128, 1, 32, 32)
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # (32, 64, 1, 64, 64)
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # (32, 32, 2, 128, 128)
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # (32, 16, 4, 256, 256)
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 4, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # (32, 3, 8, 512, 512)
            nn.Sigmoid()  # For normalized pixel values between [0, 1]
        )
    def forward(self, x):
        x = self.encoder(x)

        encode = self.bottleneck(x)

        x = self.decoder(encode)

        return encode,x


class gate(nn.Module):
    def __init__(self):
        super(gate, self).__init__()
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def forward(self, x1,x2,x3):
        x1 = F.relu(self.lin1(x1))
        x2 = F.relu(self.lin1(x2))
        x3 = F.relu(self.lin1(x3))
        x1 = F.relu(self.lin2(x1))
        x2 = F.relu(self.lin2(x2))
        x3 = F.relu(self.lin2(x3))
        x1 = F.relu(self.lin3(x1))
        x2 = F.relu(self.lin3(x2))
        x3 = F.relu(self.lin3(x3))
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
        x = torch.cat((x1,x2,x3),dim=1)
        x = self.softmax(x)
        return x


class mlp_pre(torch.nn.Module):
    def __init__(self, num_in ,num_hid1 , num_hid2 ,num_hid3,num_hid4 ,num_out ):
        super(mlp_pre, self).__init__()
        self.l1 = torch.nn.Linear(num_in, num_hid1)
        self.l2 = torch.nn.Linear(num_hid1, num_hid2)
        self.l3 = torch.nn.Linear(num_hid2, num_hid3)
        self.l4 = torch.nn.Linear(num_hid3, num_hid4)
        self.classify = torch.nn.Linear(num_hid4, num_out)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(0.5)
        self.nor = torch.nn.BatchNorm1d(num_hid1)
        self.nor2 = torch.nn.BatchNorm1d(num_hid2)
        self.nor3 = torch.nn.BatchNorm1d(8)
        self.nor4 = torch.nn.BatchNorm1d(4)
        
        # self.nor2 = torch.nn.BatchNorm1d(num_hid2)
    def forward(self, x):
        
        x = self.l1(x)

        x = self.drop(x)
        x = self.l2(x)

        x = self.drop(x)
        x = self.l3(x)


        return x


    
class MolVisGNN(nn.Module):
    def __init__(self,):
        super(MolVisGNN, self).__init__()
        self.conv1 = HeteroConv({
            ('Protein', 'interacts', 'drug'): SAGEConv(256, 128),
            ('drug', 'interacts', 'Protein'): SAGEConv(256, 128)
        },aggr='mean')
        self.conv2 = HeteroConv({
            ('Protein', 'interacts', 'drug'): SAGEConv(128, 64),
            ('drug', 'interacts', 'Protein'): SAGEConv(128, 64)
        },aggr='mean')

        self.conv3 = HeteroConv({
            ('Protein', 'interacts', 'drug'): SAGEConv(64, 32),
            ('drug', 'interacts', 'Protein'): SAGEConv(64, 32)
        },  aggr='mean')

        self.Autoencoder = Autoencoder()

        self.autp = Autoencoder()

        self.nor = torch.nn.BatchNorm1d(256)
        self.nor2 = torch.nn.BatchNorm1d(64)
        self.nor3 = torch.nn.BatchNorm1d(128)
        self.resnet = Conv1dNetwork()
        self.relu = torch.nn.LeakyReLU(0.5)
        self.gate = gate()
        self.mlp_pre = mlp_pre(64,32,16,1,1,1)

    def forward(self, x_dict, edge_index_dict,drug_2d_features, drug_3d_features):
        encode_3d = drug_3d_features
        encode_3d = self.sp(encode_3d)
        encode_3d = encode_3d.mean(dim=[2, 3, 4])
        encode_3d = F.adaptive_avg_pool1d(encode_3d, 256)
        
        encode_3d = drug_3d_features

        drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 128).squeeze(0)
        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'], 128).squeeze(0)
        drug_1d_features = x_dict['drug']

        x = self.gate(x_dict['drug'],drug_2d_features, encode_3d)
        self.saved_x = x.detach().cpu().numpy() 
        drug_2d_features = drug_2d_features * x[:,1].unsqueeze(1)
        drug_1d_features = drug_1d_features * x[:,0].unsqueeze(1)
        encode_3d = encode_3d * x[:,2].unsqueeze(1)

        x_dict['drug'] = torch.cat((drug_1d_features,drug_2d_features,encode_3d), dim=1)

        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'].unsqueeze(0), 128).squeeze(0)
        x_dict['drug'] = self.nor3(x_dict['drug'])

        drug_res = self.resnet(x_dict['drug'])


        x_dict['drug'] = drug_res + x_dict['drug']



        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'], 256).squeeze(0)
        x_dict['Protein'] = F.adaptive_avg_pool1d(x_dict['Protein'].unsqueeze(0), 256).squeeze(0)


        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key:self.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key:self.relu(x) for key, x in x_dict.items()}

        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key:self.relu(x) for key, x in x_dict.items()}



        return x_dict


    def compute_loss(self,out, batch):

        # 获取边
        edge_index = batch[('drug', 'interacts', 'Protein')].edge_label_index
        # 标签 
        labels =  batch[('drug', 'interacts', 'Protein')].edge_label
        scoreout = []
        for d,m in zip(edge_index[0],edge_index[1]) :
            
            Protein_feature = out['Protein'][m]
            drug_feature = out['drug'][d]
            edge_feature = torch.cat((drug_feature, Protein_feature ), dim=0)


            scoreout.append(edge_feature)
        scoreout = torch.stack(scoreout)

        scoreout = self.mlp_pre(scoreout)

        edge = edge_index
    
        scores = scoreout.to(device)
        labels = labels.to(device)
        scores = scores.squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels.float())


        total_loss =  loss 

        return total_loss,scores, labels,edge

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

