import torch
from torch_geometric.data import Dataset
import os
 
class BuildingDataset(Dataset):
    def __init__(self, processed_dir='data/processed/'):
        super().__init__()
        self.file_list = sorted([
            os.path.join(processed_dir, f)
            for f in os.listdir(processed_dir) if f.endswith('.pt')
        ])
 
    def len(self):
        return len(self.file_list)
 
    def get(self, idx):
        return torch.load(self.file_list[idx], weights_only=True)