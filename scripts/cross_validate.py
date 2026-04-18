import torch
import numpy as np
from aerognn.data.dataset import BuildingDataset
from aerognn.training.trainer import cross_validation
from aerognn.training.pinn_trainer import cross_validation_pinn

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def cv_GNN():
    dataset = BuildingDataset()
    mae, r2 = cross_validation(dataset, 300)
    print(f'MAE: {mae}, R^2: {r2}')

def cv_PINN():
    dataset = BuildingDataset()
    results = cross_validation_pinn(dataset, 300)
    print(f"Final R2 of scores: {results['avg_scores_r2']:.4f}")
    print(f"Final MAE of scores: {results['avg_scores_mae']:.4f}")
    print(f"Final R2 for Vx: {results['avg_vx_r2']:.4f}")
    print(f"Final MAE for Vx: {results['avg_vx_mae']:.4f}")
    print(f"Final R2 for Vy: {results['avg_vy_r2']:.4f}")
    print(f"Final MAE for Vy: {results['avg_vy_mae']:.4f}")
    print(f"Final R2 for Vz: {results['avg_vz_r2']:.4f}")
    print(f"Final MAE for Vz: {results['avg_vz_mae']:.4f}")
    print(f"Final R2 for Pressure: {results['avg_pres_r2']:.4f}")
    print(f"Final MAE for Pressure: {results['avg_pres_mae']:.4f}")

if __name__ == "__main__":
    cv_PINN()
