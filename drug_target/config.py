import torch

# 选择 GPU 设备（你之前可用的 GPU 是 0、2、3）
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
