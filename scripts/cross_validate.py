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
    set_seed(42)
    dataset = BuildingDataset()
    results = cross_validation_pinn(dataset, 300)
    print(f"Final Score R2: {results['avg_score_r2']:.4f}")
    print(f"Final Score MAE: {results['avg_score_mae']:.4f}")
    print(f"Final Cd R2: {results['avg_cd_r2']:.4f}")
    print(f"Final Cd MAE: {results['avg_cd_mae']:.4f}")
    print(f"Final Cl R2: {results['avg_cl_r2']:.4f}")
    print(f"Final Cl MAE: {results['avg_cl_mae']:.4f}")
    print(f"Final Clstd R2: {results['avg_clstd_r2']:.4f}")
    print(f"Final Clstd MAE: {results['avg_clstd_mae']:.4f}")
    print(f"Final Vx R2: {results['avg_vx_r2']:.4f}")
    print(f"Final Vx MAE: {results['avg_vx_mae']:.4f}")
    print(f"Final Vy R2: {results['avg_vy_r2']:.4f}")
    print(f"Final Vy MAE: {results['avg_vy_mae']:.4f}")
    print(f"Final Vz R2: {results['avg_vz_r2']:.4f}")
    print(f"Final Vz MAE: {results['avg_vz_mae']:.4f}")
    print(f"Final Pressure R2: {results['avg_pres_r2']:.4f}")
    print(f"Final Pressure MAE: {results['avg_pres_mae']:.4f}")

if __name__ == "__main__":
    cv_PINN()