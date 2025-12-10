import torch.nn as nn
from torch_geometric.nn import GCNConv,HeteroConv,SAGEConv,GraphConv,GATConv,GraphNorm,TAGConv
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
        # x = F.relu(self.conv3(x)) 
        # x = self.pool(x)  
        x = x.view(x.size(0), -1) 
        x = self.pool2(x)
        x = x.squeeze(1)
        # x = self.nor(x)
        return x
    



    



class gat(nn.Module):
    def __init__(self):
        super(aaaa, self).__init__()
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
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
        # x1 = self.sigmoid(x1)
        # x2 = self.sigmoid(x2)
        # x3 = self.sigmoid(x3)
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
    def __init__(self, num_in ,num_hid1 , num_hid2 , num_out):
        super(mlp_pre, self).__init__()
        self.l1 = torch.nn.Linear(num_in, num_hid1)
        self.l2 = torch.nn.Linear(num_hid1, num_hid2)
        self.classify = torch.nn.Linear(num_hid2, num_out)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(0.5)
        self.nor = torch.nn.BatchNorm1d(num_hid1)
        self.nor2 = torch.nn.BatchNorm1d(num_hid2)
    def forward(self, x):
        
        x = self.l1(x)
        x = self.nor(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.nor2(x)
        x = self.classify(x)

        return x,0



class Directional3DProcessor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Directional3DProcessor, self).__init__()

        self.conv_fr = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_bb = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_tl = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),

        )

    def forward(self, encoded_3d):  # [B, 128, 6, 32, 32]
        fr = encoded_3d[:, :, 0:2]  # [B, 128, 2, 32, 32]
        bb = encoded_3d[:, :, 2:4]
        tl = encoded_3d[:, :, 4:6]

        fr_out = self.conv_fr(fr)
        bb_out = self.conv_bb(bb)
        tl_out = self.conv_tl(tl)


        combined = torch.cat([fr_out, bb_out, tl_out], dim=2)  
        return combined  # shape: [B, out_channels, 6, 32, 32]




class MolVisGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_channels2,out_channels_gat, 
                  out_channels, global_dim, num_layers, heads, ff_dropout, 
                 attn_dropout, spatial_size, skip, dist_count_norm, conv_type,num_centroids, no_bn, norm_type):
        super(zhangzimai, self).__init__()
        self.conv1 = HeteroConv({
            ('miRNA', 'interacts', 'drug'): SAGEConv(256, 128),
            ('drug', 'interacts', 'miRNA'): SAGEConv(256, 128)
        },aggr='mean')
        self.conv2 = HeteroConv({
            ('miRNA', 'interacts', 'drug'): SAGEConv(512, 256),
            ('drug', 'interacts', 'miRNA'): SAGEConv(512, 256)
        },aggr='mean')
        self.conv3 = HeteroConv({
            ('miRNA', 'interacts', 'drug'): SAGEConv(256, 128),
            ('drug', 'interacts', 'miRNA'): SAGEConv(256, 128)
        },  aggr='mean')
        self.conv4 = HeteroConv({
            ('miRNA', 'interacts', 'drug'): SAGEConv(32, 16),
            ('drug', 'interacts', 'miRNA'): SAGEConv(32, 16)
        },  aggr='mean')
        self.dropout = nn.Dropout(0.5)  
        self.dropout_3d = nn.Dropout(0.3)
        self.dropout_feat = nn.Dropout(0.4)  

        self.mlp_pre = mlp_pre(256,128,64,1)
        self.mlp_pre2 = mlp_pre2(512,512,256,1)
        self.lne = torch.nn.Linear(382, 256)
        self.nor3d = torch.nn.LayerNorm(128)
        self.nor = torch.nn.LayerNorm(128)
        self.nor2 = torch.nn.LayerNorm(256)
        self.nor3 = torch.nn.BatchNorm1d(128)
        
        self.resnet = Conv1dNetwork()
        self.relu = torch.nn.LeakyReLU(0.01)
        self.gat = gat()
        self.sp = Directional3DProcessor(128,32)
        self.norm_drug_1 = GraphNorm(256)
        self.norm_miRNA_1 = GraphNorm(256)
        self.norms = nn.ModuleDict({
            'drug': GraphNorm(256),
            'miRNA': GraphNorm(256),
        })
    def forward(self, x_dict, edge_index_dict,drug_2d_features, drug_3d_features):
        # print(drug_3d_features.shape)
        # encode_3d = drug_3d_features
        # print(drug_3d_features.shape)


        encode_3d = self.sp(drug_3d_features)
        encode_3d = self.dropout_3d(encode_3d)
        encode_3d_S = encode_3d.mean(dim=[2, 3, 4])
        encode_3d_s2 = encode_3d.mean(dim=[2, 3, 4])
        encode_3d = F.adaptive_avg_pool1d(encode_3d_S, 128)
        encode_3d2 = F.adaptive_avg_pool1d(encode_3d_s2, 256)




        drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 128)
        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'], 128)
        drug_1d_features = x_dict['drug']

        
        x = self.gat(x_dict['drug'],drug_2d_features, encode_3d)
        drug_2d_features = drug_2d_features * x[:,1].unsqueeze(1)
        drug_1d_features = drug_1d_features * x[:,0].unsqueeze(1)
        encode_3d = encode_3d * x[:,2].unsqueeze(1)
        x_dict['drug'] = torch.cat((drug_1d_features,drug_2d_features,encode_3d), dim=1)
        x_dict['drug'] = self.dropout_feat(x_dict['drug']) 
        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'], 128)

        x_dict['drug'] = self.nor3d(x_dict['drug'])
        drug_res = self.resnet(x_dict['drug'])


        x_dict['drug'] = drug_res + x_dict['drug']

        xdrug = F.adaptive_avg_pool1d(x_dict['drug'], 128)
        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'], 256)
        

        xmi = x_dict['miRNA']=F.adaptive_avg_pool1d(x_dict['miRNA'], 128)
        x_dict['miRNA'] = F.adaptive_avg_pool1d(x_dict['miRNA'], 256)
        


        drug_feat = x_dict['drug'] 
        mirna_feat = x_dict['miRNA']  
        combined_feat = torch.cat([drug_feat, mirna_feat], dim=0)  

        combined_feat = F.normalize(combined_feat, p=2, dim=-1)  
        

        N_drug = drug_feat.size(0)
        drug_feat_norm = combined_feat[:N_drug, :]  
        mirna_feat_norm = combined_feat[N_drug:, :]  
        
        x_dict['drug'] = drug_feat_norm
        x_dict['miRNA'] = mirna_feat_norm


        x_dict = self.conv1(x_dict, edge_index_dict)
        for ntype, x in x_dict.items():
            x_dict[ntype] = self.nor(self.relu(x))
        x_dict['drug'] = xdrug+x_dict['drug']
        x_dict['miRNA'] = xmi+x_dict['miRNA']


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

        # 损失计算
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels.float())

        total_loss = loss

        return total_loss, scores, labels, edge_index,t




    
    def test(self, output, label):
        positive_class_probs = torch.sigmoid(output).detach().cpu().numpy()
        targets = label.cpu().numpy()


        auc = roc_auc_score(targets, positive_class_probs)
        aupr = average_precision_score(targets, positive_class_probs)

        predicted = (positive_class_probs > 0.5).astype(int)


        accuracy = accuracy_score(targets, predicted)
        precision = precision_score(targets, predicted, zero_division=0)
        recall = recall_score(targets, predicted, zero_division=0)
        f1 = f1_score(targets, predicted, zero_division=0)

        return auc, aupr, accuracy, precision, recall, f1



