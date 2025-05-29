import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import Dataset, DataLoader
import pyvista as pv  

# ========= 环境设置 =========
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# ========= 模型定义 =========
class ConvAutoencoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, stride=(1,2,2), padding=1),
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1,2,2), padding=1),
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1,2,2), padding=1),
            nn.ReLU(True),
            nn.Conv3d(64,128, kernel_size=3, stride=(1,2,2), padding=1),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=(1,2,2),
                               padding=1, output_padding=(0,1,1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=(1,2,2),
                               padding=1, output_padding=(0,1,1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(1,2,2),
                               padding=1, output_padding=(0,1,1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(16,  4, kernel_size=3, stride=(1,2,2),
                               padding=1, output_padding=(0,1,1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)



class AllInMemoryDataset(Dataset):
    def __init__(self, folder_path, device):
        self.paths = [os.path.join(folder_path, f)
                      for f in os.listdir(folder_path) if f.endswith('.npy')]
        self.device = device
        self.data = []
        print("Loading all .npy files into GPU...")
        for p in tqdm(self.paths):
            arr = np.load(p)              # shape [4,6,H,W] or [6,4,H,W]
            t   = torch.from_numpy(arr).float().to(device)
            self.data.append(t)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.paths[idx]



def visualize_original_views(tensor4d, save_dir, name="sample0"):
    if tensor4d.ndim == 4 and tensor4d.shape[0] == 6:
        tensor4d = tensor4d.permute(1, 0, 2, 3)
    assert tensor4d.shape[:2] == (4, 6)
    os.makedirs(save_dir, exist_ok=True)
    vol = tensor4d[0]  # [6,H,W]
    fig, axs = plt.subplots(1, 6, figsize=(18, 3))
    def norm(img):
        mi, ma = img.min(), img.max()
        return np.clip((img - mi) / (ma - mi + 1e-8), 0, 1)
    for v in range(6):
        img = vol[v].cpu().numpy()
        axs[v].imshow(norm(img), cmap='gray')
        axs[v].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=200)
    plt.close()



def save_encoded_features(model, loader, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for x, paths in tqdm(loader, desc="Saving Encoded Features"):
            if x.shape[1]==6 and x.shape[2]==4:
                x = x.permute(0,2,1,3,4)
            x = x.to(device)
            encoded, _ = model(x)
            for i, p in enumerate(paths):
                name = os.path.splitext(os.path.basename(p))[0] + "_encoded.npy"
                np.save(os.path.join(save_dir, name), encoded[i].cpu().numpy())


def train_autoencoder(model, loader, optimizer, device, epochs):
    model.to(device)
    loss_fn = nn.MSELoss()
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        for x, _ in tqdm(loader, desc=f"Epoch {ep}/{epochs}"):
            if x.shape[1]==6 and x.shape[2]==4:
                x = x.permute(0,2,1,3,4)
            x = x.to(device)
            _, xr = model(x)
            loss = loss_fn(xr, x)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        print(f"Epoch {ep}, Loss: {total/len(loader):.6f}")

        if ep % 10 == 0:
            xb, paths = next(iter(loader))
            if xb.shape[1]==6 and xb.shape[2]==4:
                xb = xb.permute(0,2,1,3,4)
            xb = xb.to(device)
            with torch.no_grad():
                _, xr = model(xb)



def main():
    npy_dir   = "graph_DDI/data/ddi_npy"
    vis_o     = "graph_DDI/vis_original"
    enc_dir   = "graph_DDI/data/ddi_encoded"
    w_dir     = "graph_DDI/3dmodel_weights"
    os.makedirs(w_dir, exist_ok=True)

    ds     = AllInMemoryDataset(npy_dir, device)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    sample, _ = ds[0]
    visualize_original_views(sample, vis_o, name="orig0")

    model     = ConvAutoencoder1()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Starting training...")
    train_autoencoder(model, loader, optimizer, device, epochs=100)

    print("Saving encoded features...")
    save_encoded_features(model, loader, device, enc_dir)


if __name__ == "__main__":
    main()
