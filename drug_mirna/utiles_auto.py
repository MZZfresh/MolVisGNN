import csv
import torch
import pandas as pd
import numpy as np
import os

from config import device


def get_mirna_id(mirna_name1):  
    file_path = 'unique_ncrna_miRBase.csv'
    with open(file_path, 'r') as file:  
        reader = csv.reader(file)
        next(reader)  
        for row in reader:  
            mirna_name, mirna_id = row  
            if mirna_name == mirna_name1:  
                return mirna_id  
            
def get_mirna_features(mirna_name1):  
    mirna_id = get_mirna_id(mirna_name1)   
    file_path = 'kmer_features.csv'  
    with open(file_path, 'r') as file:  
        for line in file:  
            parts = line.strip().split(',')  
            if parts and parts[0] == mirna_id:  

                numeric_features = [float(feat) for feat in parts[1:]]  

                features_tensor = torch.tensor(numeric_features, dtype=torch.float32)  
                return features_tensor  

def lode1d_to_gpu(file_path,device):
    csv_file = file_path  
    df = pd.read_csv(csv_file,header=None)
    df = df.fillna(0)

    drug_dict = {row[0]: pd.to_numeric(row[1:], errors='coerce') for row in df.values}

    gpu_feature_dict = {}

    for drug, features in drug_dict.items():

        feature_tensor = torch.tensor(features, dtype=torch.float32)
        

        feature_tensor = feature_tensor.to(device)

        gpu_feature_dict[drug] = feature_tensor
    print(f" Features 1d on GPU: {csv_file}")
    return gpu_feature_dict

def load_mirna_features(file_path,device):
    csv_file = file_path  
    df = pd.read_csv(csv_file)
    mirna_dict = {row[0]: pd.to_numeric(row[1:], errors='coerce') for row in df.values} 
    gpu_feature_dict = {}

    for mirna_id, features in mirna_dict.items():

        feature_tensor = torch.tensor(features, dtype=torch.float32)

        feature_tensor = feature_tensor.to(device)

        gpu_feature_dict[mirna_id] = feature_tensor
    print(f" Features mirna on GPU: {csv_file}")
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
    with open('drug_mapping.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)   
        for row in reader:  
            drug_name1, drug_id = row  

            if drug_id == drug_idex:

                return drug_name1
            



def load_npy_to_gpu(npy_files, device):
    gpu_data = {}
    for file_path in npy_files:
        # Load the .npy file as a numpy array
        np_array = np.load(file_path)
        # Convert the numpy array to a torch tensor
        tensor = torch.from_numpy(np_array)
        # Move the tensor to the specified device (GPU)
        tensor = tensor.to(device).type(torch.float32)
        # Get the file name without path
        file_name = os.path.basename(file_path)
        # Add the tensor to the dictionary with the file name as the key
        gpu_data[file_name] = tensor
    return gpu_data
from einops import rearrange
def to_3d(x):
    return rearrange(x, '  c d h w -> (c d) h w ')

def get_drug_features(data,gpu_data_3d):
    drug_features_3d_tensors = []
    for idex in data:  
        b = get_drug_name(str(idex.item()))
        a = get_drug_id(b)   
        key = a+".npy"
        drug_features = gpu_data_3d.get(key)


        new_num_features = 8   
        additional_features = new_num_features - drug_features.size(1)  

        zero_padding = torch.zeros(drug_features.size(0),additional_features, drug_features.size(2), drug_features.size(3), dtype=torch.float32).to(device) 


        drug_3d_features = torch.cat((drug_features, zero_padding), dim=1).to(device) 

        drug_features_3d_tensors.append(drug_3d_features) 

    stacked_3d_tensor = torch.stack(drug_features_3d_tensors, dim=0)  
    return stacked_3d_tensor

def get_drug_2d_features(data,gpu_2d):
    drug_features_2d_tensors = []
    for idex in data:  

        b = get_drug_name(str(idex.item()))

        drug_features = gpu_2d.get(b)
        # print(drug_features)
        drug_features_2d_tensors.append(drug_features)

    stacked_2d_tensor = torch.stack(drug_features_2d_tensors, dim=0)  

    return stacked_2d_tensor

