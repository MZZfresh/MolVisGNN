from get_graph_auto import *
from torch_geometric.loader import HGTLoader,neighbor_loader,NeighborLoader,LinkNeighborLoader
from graph_model_auto import zhangzimai,Autoencoder
from graph_model_auto import *
import argparse
import gc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import device
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") 
# 输出设备名称  
print(f"Using device: {device}")  

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2000, help="Random seed for model and dataset.")
parser.add_argument('--epochs', type=int, default=1500, help="Number of epochs to train.")
parser.add_argument('--lr', type=float, default=0.0004, help='learning rate in optimizer')
parser.add_argument('--wd', type=float, default=0.000000001, help='weight decay in optimizer')
args = parser.parse_args()


best_val_loss = float('inf') 

def set_random_seed(seed):
    np.random.seed(seed)
    # 固定PyTorch的随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 关键补充：确保 cuDNN 可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False







def test(test_data,model,gpu_2d,gpu_data_3d,best_model_state_dict):
    model.eval()
    model.load_state_dict(best_model_state_dict) 
    test_scores = []
    node_id_mapping = {}
    model.eval()
    with torch.no_grad():


        drug_2d_features_val = get_drug_2d_features2(test_data['drug'].node_idx.tolist(), gpu_2d).to(device)
        drug_3d_features_val = get_drug_features3d(test_data['drug'].node_idx.tolist(), gpu_data_3d).to(device)

        out_val = model(test_data.x_dict, test_data.edge_index_dict,
                                    drug_2d_features_val, drug_3d_features_val)

        val_loss,val_scores, val_labels,edge_index = model.compute_loss(out_val, test_data)
        auc_val,aupr_val,accuracy_val,precision_val,recall_val ,f1_val= model.test(val_scores, val_labels)



        results = {
            "avg_val_auc": [auc_val],
            "avg_val_aupr": [aupr_val],
            "avg_val_accuracy": [accuracy_val],
            "avg_val_precision": [precision_val],
            "avg_val_recall": [recall_val],
            "avg_val_f1": [f1_val],

        }

        df_results = pd.DataFrame(results)

        positive_class_probs = F.sigmoid(val_scores)
    
        for score, label, orig_edge in zip(positive_class_probs, val_labels, edge_index.T):  
            test_scores.append({
                'score': score.item(),
                'label': label.item(),
                'drug_node_id': orig_edge[1].item(),  
                'miRNA_node_id': orig_edge[0].item()  
            })
        print(results)
    return 0




def get_drug_2d_features2(data,gpu_2d):
    drug_features_2d_tensors = []
    for idex in data:  

        b = get_drug_name(str(int(idex)))
        drug_features = gpu_2d.get(b)
        # print(drug_features)
        drug_features_2d_tensors.append(drug_features)

    stacked_2d_tensor = torch.stack(drug_features_2d_tensors, dim=0)  

    return stacked_2d_tensor

def get_drug_features3d(node_idxs, drug_features_dict):
    drug_feature = []

    for node_idx in node_idxs:
        drug_name = get_drug_name(str(int(node_idx)))
        drug_id = get_drug_id(drug_name)
        drug_name = drug_id+ '_output_encoded.npy'

        drug_features = drug_features_dict.get(drug_name, None)
        if drug_features is not None:

            drug_feature.append(drug_features)

        else:
            print(f"No features found for {drug_name}. 3dSkipping.")

            continue


    drug_feature = torch.stack(drug_feature, dim=0)
    return drug_feature



def main():
    seed = args.seed
    set_random_seed(seed)

    npy_files_dir = 'dmi_encodedre'  
    npy_files = [os.path.join(npy_files_dir, file) for file in os.listdir(npy_files_dir) if file.endswith('.npy')]  
    gpu_data_3d = load_npy_to_gpu(npy_files, device) 

    gpu_1d = lode1d_to_gpu('drug_1d_features.csv',device=device)
    gpu_2d = lode1d_to_gpu('128_2d.csv',device=device)
    mirna_1d_feature = load_mirna_features('kmer_features.csv',device=device)   


    train_data, val_data, test_data= get_graph(gpu_1d,mirna_1d_feature)


    model = zhangzimai()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    best_auc = 999999999
    model.to(device)
    best_model_state_dict = None

    edge_type = ('drug', 'interacts', 'miRNA')
    edge_index_tensor = train_data[edge_type].edge_label_index  
    edge_labels = train_data[edge_type].edge_label  


    best_val_loss = float('inf') 



    pbar_epochs = tqdm(total=args.epochs, desc="Training")
    for epoch in range(args.epochs):
        model.train()
        drug_2d_features = get_drug_2d_features2(train_data['drug'].node_idx.tolist(),gpu_2d)
        drug_3d_features = get_drug_features3d(train_data['drug'].node_idx.tolist(),gpu_data_3d)
        out = model(train_data.x_dict, train_data.edge_index_dict,drug_2d_features,drug_3d_features)
        
        loss,scores, labels,teain_edge,_ = model.compute_loss(out, train_data)


        optimizer.zero_grad()



        loss.backward()




        optimizer.step()
        scheduler.step()

        pbar_epochs.update(1)

        if epoch % 50 == 0:
            print(epoch)
            
            model.eval()
            with torch.no_grad:
                drug_2d_features_val = get_drug_2d_features2(val_data['drug'].node_idx.tolist(), gpu_2d).to(device)
                drug_3d_features_val = get_drug_features3d(val_data['drug'].node_idx.tolist(), gpu_data_3d).to(device)

                out_val = model(val_data.x_dict, val_data.edge_index_dict,
                                            drug_2d_features_val, drug_3d_features_val)

                val_loss,val_scores, val_labels,edge,t= model.compute_loss(out_val, val_data)

                auc_val,aupr_val,accuracy_val,precision_val,recall_val ,f1_val= model.test(val_scores, val_labels)


                if val_loss < best_auc:
                    best_auc = val_loss
                    best_model_state_dict = model.state_dict()
                    # torch.save(best_model_state_dict, 'graph/model_path/Auto_best_model_ronghe_model.pth')
                    # torch.save(model, 'graph/model_path/Auto_best_model_ronghe.pth')

                results = {
                    "avg_val_auc": [auc_val],
                    "avg_val_aupr": [aupr_val],
                    "avg_val_accuracy": [accuracy_val],
                    "avg_val_precision": [precision_val],
                    "avg_val_recall": [recall_val],
                    "avg_val_f1": [f1_val],
                }
                df = pd.DataFrame(results)
                print(results)


    test(test_data,model,gpu_2d,gpu_data_3d,best_model_state_dict)

main()



