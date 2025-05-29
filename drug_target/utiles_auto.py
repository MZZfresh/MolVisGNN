import csv
import torch
import pandas as pd
import numpy as np
import os

from config import device



def get_Protein_id(Protein_name1):  
    file_path = 'graph/data/unique_ncrna_miRBase.csv'
    with open(file_path, 'r') as file:  
        reader = csv.reader(file)
        next(reader)   
        for row in reader:  
            Protein_name, Protein_id = row  
            if Protein_name == Protein_name1:  
                return Protein_id  
            
def get_Protein_features(Protein_name1):  
    Protein_id = get_Protein_id(Protein_name1)   
    file_path = 'graph-DTI/kmer_features.csv'  
    with open(file_path, 'r') as file:  
        for line in file:  
            parts = line.strip().split(',')  
            if parts and parts[0] == Protein_id:  

                numeric_features = [float(feat) for feat in parts[1:]]  

                features_tensor = torch.tensor(numeric_features, dtype=torch.float32)  
                return features_tensor 

def lode1d_to_gpu(file_path,device):
    csv_file = file_path  
    df = pd.read_csv(csv_file,header=None)
    df = df.fillna(0)

    drug_dict = {int(row[0]): pd.to_numeric(row[1:], errors='coerce') for row in df.values}

    gpu_feature_dict = {}

    for drug, features in drug_dict.items():

        feature_tensor = torch.tensor(features, dtype=torch.float32)
        

        feature_tensor = feature_tensor.to(device)

        gpu_feature_dict[drug] = feature_tensor
    print(f" Features 1d on GPU: {csv_file}")
    return gpu_feature_dict

def load_Protein_features(file_path,device):
    csv_file = file_path 
    df = pd.read_csv(csv_file)
    Protein_dict = {row[0]: pd.to_numeric(row[1:], errors='coerce') for row in df.values} 
    gpu_feature_dict = {}

    for Protein_id, features in Protein_dict.items():

        feature_tensor = torch.tensor(features, dtype=torch.float32)

        feature_tensor = feature_tensor.to(device)

        gpu_feature_dict[Protein_id] = feature_tensor
    print(f" Features Protein on GPU: {csv_file}")
    return gpu_feature_dict

def get_drug_id(drug_name):
    with open('graph/data/drug_list.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)   
        for row in reader:  
            drug_name1, drug_id = row  
            if drug_name1 == drug_name:  
                return drug_id


def get_drug_name(drug_idex):
    with open('drug_index.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:  
            drug_name1, drug_id = row  
            if drug_idex == drug_id:  
                return drug_name1

def load_npy_to_gpu(npy_files, device):
    gpu_data = {}
    for file_path in npy_files:

        np_array = np.load(file_path)

        tensor = torch.from_numpy(np_array)

        tensor = tensor.to(device).type(torch.float32)

        file_name = os.path.basename(file_path)

        gpu_data[file_name] = tensor
    return gpu_data
from einops import rearrange
def to_3d(x):
    return rearrange(x, '  c d h w -> (c d) h w ')

def get_drug_features(data,gpu_data_3d):
    drug_features_3d_tensors = []
    for idex in data:  
        b = get_drug_name(str(idex.item()))

        key = b+"_output.npy"
        drug_features = gpu_data_3d.get(key)
        drug_features = drug_features.permute(1,0,2,3)

        drug_features_3d_tensors.append(drug_features) 

    stacked_3d_tensor = torch.stack(drug_features_3d_tensors, dim=0)  
    return stacked_3d_tensor

def get_drug_2d_features(data,gpu_2d):
    drug_features_2d_tensors = []

    for idex in data:  
        
        b = get_drug_name(str(idex.item()))
        
        b = int(b)
        drug_features = gpu_2d.get(b)

        if drug_features is None:
            print(f"Warning: Drug ID {b} has no corresponding 2D feature!")
        drug_features_2d_tensors.append(drug_features)

    stacked_2d_tensor = torch.stack(drug_features_2d_tensors, dim=0)  

    return stacked_2d_tensor