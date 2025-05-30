from get_graph import *
from torch_geometric.loader import NeighborSampler,GraphSAINTRandomWalkSampler
from graph_model import zhangzimai,Autoencoder
from graph_model import *
import argparse
from tqdm import tqdm
from torch_geometric.data import Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from config import device
import random
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

print(f"Using device: {device}")  
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2000, help="Random seed for model and dataset.")
parser.add_argument('--epochs', type=int, default=300, help="Number of epochs to train.")
parser.add_argument('--lr', type=float, default=0.006, help='learning rate in optimizer')
parser.add_argument('--wd', type=float, default=0.0002, help='weight decay in optimizer')
args = parser.parse_args()


best_val_loss = float('inf') 




def get_drug_features(node_idxs, idx_to_node, drug_features_dict):
    drug_feature = []

    for node_idx in node_idxs:
        drug_name = idx_to_node.get(node_idx.item(), None)

        drug_features = drug_features_dict.get(drug_name, None)
        if drug_features is not None:

            drug_feature.append(drug_features)
        else:
            print(f"No features found for {drug_name}. Skipping.")

            continue

    drug_feature = torch.stack(drug_feature, dim=0)
    return drug_feature


def to_3d(x):
    return rearrange(x, '  c d h w -> (c d) h w ')

def get_drug_features3d(node_idxs, idx_to_node, drug_features_dict):
    drug_feature = []

    for node_idx in node_idxs:
        drug_name = idx_to_node.get(node_idx.item(), None)
        drug_name = drug_name+ '_output_encoded.npy'

        drug_features = drug_features_dict.get(drug_name, None)
        if drug_features is not None:

            drug_feature.append(drug_features)

        else:
            print(f"No features found for {drug_name}. 3dSkipping.")

            continue


    drug_feature = torch.stack(drug_feature, dim=0)
    return drug_feature





def test(model, gpu_1d, gpu_2d, gpu_data_3d, best_model_state_dict):
    model.eval()
    model.load_state_dict(best_model_state_dict) 
    test_graph, idx_to_node, node_to_idx = get_graph()

    with torch.no_grad():
        # 获取药物特征
        drug_1d_features_val = get_drug_features(test_graph.node_idx, idx_to_node, gpu_1d)
        drug_2d_features_val = get_drug_features(test_graph.node_idx, idx_to_node, gpu_2d)
        drug_3d_features_val = get_drug_features3d(test_graph.node_idx, idx_to_node, gpu_data_3d)
        out_val = model(test_graph, drug_1d_features_val, drug_2d_features_val, drug_3d_features_val)
    return 0  



def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    seed = args.seed
    set_random_seed(seed)
    npy_files_dir = 'ddi_encodedre' 
    npy_files = [os.path.join(npy_files_dir, file) for file in os.listdir(npy_files_dir) if file.endswith('.npy')]  
    gpu_data_3d = load_npy_to_gpu(npy_files, device) 
    gpu_1d = lode1d_to_gpu('graph_DDI/data/drug_1d_fingerprints.csv',device=device)
    gpu_2d = lode2d_to_gpu('graph_DDI/data/drug_2d.csv',device=device)
    train_data, val_data, test_data,idx_to_node= get_graph()
    
    model = MolVisGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
    
    
    model.to(device)
    best_model_state_dict = None
    best_val_loss = float('inf') 
    pbar_epochs = tqdm(total=args.epochs, desc=f"Training", leave=False)
    for epoch in range(args.epochs):
        model.train()  
        drug_1d_features = get_drug_features(train_data.node_idx,idx_to_node,gpu_1d)
        drug_2d_features = get_drug_features(train_data.node_idx,idx_to_node,gpu_2d)
        drug_3d_features = get_drug_features3d(train_data.node_idx,idx_to_node,gpu_data_3d)
        out,weight= model(train_data,drug_1d_features,drug_2d_features,drug_3d_features)
        loss,scores, labels, edge,_= model.compute_loss(out, train_data)
        loss = loss 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        pbar_epochs.set_postfix({
                'loss': f'{loss.item():.4f}',
            })
        pbar_epochs.update(1)
        if (epoch ) % 50 == 0:
            # model.eval()
            with torch.no_grad():
                drug_1d_features_val = get_drug_features(val_data.node_idx,idx_to_node,gpu_1d)
                drug_2d_features_val = get_drug_features(val_data.node_idx,idx_to_node,gpu_2d)
                drug_3d_features_val = get_drug_features3d(val_data.node_idx,idx_to_node,gpu_data_3d)
                out_val,weight_val = model(val_data,drug_1d_features_val,drug_2d_features_val,drug_3d_features_val)
                val_loss,val_scores,  val_labels, edge,t = model.compute_loss(out_val, val_data)
                auc_val,aupr_val,accuracy_val,precision_val,recall = model.test(val_scores, val_labels)
                print('val_loss',val_loss.item())
                print('auc_val',auc_val)
                print('accuracy_val',accuracy_val)

                if val_loss < best_val_loss:   
                    best_val_loss = val_loss  
                    best_model_state_dict = model.state_dict()  
                results = {
                    
                    "avg_val_auc": [auc_val],
                    "avg_val_aupr": [aupr_val],
                    "avg_val_accuracy": [accuracy_val],
                    "avg_val_precision": [precision_val],
                    'recall': [recall],
                }
                print(results)

main()
