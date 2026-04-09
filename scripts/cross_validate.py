import torch
import numpy as np
from aerognn.data.dataset import BuildingDataset
from aerognn.training.trainer import cross_validation

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def cv():
    dataset = BuildingDataset()
    mae, r2 = cross_validation(dataset, 300)
    print(f'MAE: {mae}, R^2: {r2}')

if __name__ == "__main__":
    cv()
