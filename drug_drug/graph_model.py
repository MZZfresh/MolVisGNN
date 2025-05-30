import torch.nn as nn
from torch_geometric.nn import GCNConv,HeteroConv,SAGEConv,GraphConv,GATConv
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,roc_curve, auc,precision_recall_curve,average_precision_score,f1_score,recall_score
from config import device



class Conv1dNetwork(nn.Module):
    def __init__(self):
        super(Conv1dNetwork, self).__init__()
        # 第一层卷积
        self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1)
        # 第二层卷积
        self.conv2 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)
        # 第三层卷积
        self.conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool1d(4)
        self.pool2 = nn.AdaptiveAvgPool1d(128)
        self.pool3 = nn.AdaptiveMaxPool1d(512)
        self.nor = nn.BatchNorm1d(256)
        

    def forward(self, x):
        x = x.unsqueeze(1)  
        

        x = F.relu(self.conv1(x))  
        x = self.pool(x)  

        
        x = F.relu(self.conv2(x))  
        x = self.pool(x)  
        
        x = F.relu(self.conv3(x))  
        x = self.pool(x)  


        x = x.mean(dim=[1, 2])
        x = x.unsqueeze(1)
        x = self.pool2(x)
        x = x.squeeze(1)
        return x





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













class conv2d(nn.Module):
    def __init__(self):
        super(conv2d, self).__init__()
        self.conv1 = nn.Conv2d(24, 16, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):

        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)

        x = self.pool2(x)

        x = self.conv3(x)

        x = self.pool3(x)


        return x





class mlp_pre(torch.nn.Module):
    def __init__(self, num_in ,num_hid1 , num_hid2 ,num_hid3 ):
        super(mlp_pre, self).__init__()
        self.l1 = torch.nn.Linear(num_in, num_hid1)
        self.l2 = torch.nn.Linear(num_hid1, num_hid2)
        self.l3 = torch.nn.Linear(num_hid2, num_hid3)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(0.5)
        self.nor = torch.nn.BatchNorm1d(32)
        self.nor2 = torch.nn.BatchNorm1d(16)
        self.nor3 = torch.nn.BatchNorm1d(8)
        self.nor4 = torch.nn.BatchNorm1d(4)
        
        # self.nor2 = torch.nn.BatchNorm1d(num_hid2)
    def forward(self, x):
        
        x = self.l1(x)
        x = self.nor(x)
        x = self.relu(x)
        x = self.drop(x)
        x2 = self.l2(x)
        x = self.l2(x)
        x = self.drop(x)
        x = self.nor2(x)
        x = self.relu(x) 
        x = self.l3(x)


        return x,x2
    

class Directional3DProcessor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Directional3DProcessor, self).__init__()

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

    def forward(self, encoded_3d):  # [B, 128, 6, 32, 32]
        fr = encoded_3d[:, :, 0:2]  # [B, 128, 2, 32, 32]
        bb = encoded_3d[:, :, 2:4]
        tl = encoded_3d[:, :, 4:6]

        fr_out = self.conv_fr(fr)
        bb_out = self.conv_bb(bb)
        tl_out = self.conv_tl(tl)


        combined = torch.cat([fr_out, bb_out, tl_out], dim=2) 
        return combined  

class MolVisGNN(nn.Module):
    def __init__(self,):
        super(MolVisGNN, self).__init__()

        self.conv1 = SAGEConv(128, 128)
        self.conv2 = SAGEConv(128, 64)
        self.conv3 = SAGEConv(64, 32)
        self.sp = Directional3DProcessor(128,64)
        self.con = conv2d()
        self.lne = nn.Linear(768,128)
        self.mlp_pre = mlp_pre(64,32,16,1)
        self.nor = torch.nn.BatchNorm1d(128)
        self.nor2 = torch.nn.BatchNorm1d(64)
        self.nor3 = torch.nn.BatchNorm1d(32)
        self.resnet = Conv1dNetwork()
        self.relu = torch.nn.LeakyReLU(0.3)
        self.gate = gate()
        self.intermediate_feature = None  
        self.intermediate_gradient = None  
    def forward(self, batch,drug_1d_features,drug_2d_features,drug_3d_features):
        
        encode_3d = drug_3d_features
        encode_3d = self.sp(encode_3d)
        encode_3d = encode_3d.mean(dim=[2, 3, 4])
        encode_3d = F.adaptive_avg_pool1d(encode_3d, 256)
        


        drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 256)
        drug_1d_features = F.adaptive_avg_pool1d(drug_1d_features, 256)

        weights = self.aaa(drug_1d_features,drug_2d_features, encode_3d)

        self.saved_x = weights
        w1 = weights[:,0].unsqueeze(1)
        w2 = weights[:,1].unsqueeze(1)
        w3 = weights[:,2].unsqueeze(1)
        drug_2d_features = drug_2d_features * w1
        drug_1d_features = drug_1d_features * w2
        drug_3d_features = encode_3d * w3

        drug_feature = torch.cat((drug_1d_features,drug_2d_features,drug_3d_features), dim=1)

        drug_feature = self.lne(drug_feature)

        drug_feature = self.nor(drug_feature)
        drug_res = self.resnet(drug_feature)
        drug_feature = drug_res + drug_feature

        edge_index = batch.edge_index.to(device)

        x_dict = self.conv1(drug_feature, edge_index)
        x_dict= self.nor(x_dict)
        x_dict = self.relu(x_dict)

        x_dict = self.conv2(x_dict, edge_index)
        x_dict = self.nor2(x_dict)
        x_dict = self.relu(x_dict)

        x_dict = self.conv3(x_dict,  edge_index)
        x_dict = self.nor3(x_dict)
        x_dict = self.relu(x_dict)


        return x_dict,self.saved_x

    
    def compute_loss(self, out, batch):
        edge_index = batch.edge_label_index
        labels = batch.edge_label


        src = edge_index[0]
        dst = edge_index[1]

        drug1 = out[src]  #
        drug2 = out[dst]  


        mout = torch.cat([drug1, drug2], dim=1)  


        scores, t = self.mlp_pre(mout)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores.squeeze(), labels.float().to(scores.device)
        )

        return loss, scores, labels, edge_index, t
    
    
    
    def test(self,output,label):
        

        positive_class_probs = F.sigmoid(output)

        positive_class_probs = positive_class_probs.detach().cpu().numpy()
        targets = label.cpu().numpy()


        auc = roc_auc_score(targets, positive_class_probs)

        aupr = average_precision_score(targets, positive_class_probs)
        positive_class_probs = positive_class_probs.flatten()
        targets = targets.flatten()
        positive_class_probs = (positive_class_probs > 0.5).astype(int)
        accuracy = accuracy_score(targets, positive_class_probs)


        precision = precision_score(targets, positive_class_probs)
        recall = recall_score(targets, positive_class_probs)
        return auc,aupr,accuracy,precision,recall
        
