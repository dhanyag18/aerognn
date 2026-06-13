import torch
import numpy as np
import copy
from aerognn.data.dataset import BuildingDataset
from aerognn.training.trainer import cross_validation
from aerognn.training.pinn_trainer import cross_validation_pinn
from aerognn.training.mf_trainer import cross_validation_mf
from aerognn.data.multi_fidelity_dataset import MultiFidelityDataset
from aerognn.data.interpolate import interpolate_fine_to_coarse
from aerognn.data.groups import get_coarse_to_fine

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_results(results):
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


def cv_GNN():
    dataset = BuildingDataset()
    mae, r2 = cross_validation(dataset, 300)
    print(f'MAE: {mae}, R^2: {r2}')


def cv_PINN():
    set_seed(42)
    dataset = BuildingDataset()
    results = cross_validation_pinn(dataset, 300)
    print_results(results)


def build_fine_overlap(coarse_lookup, fine_lookup, coarse_to_fine):
    fine_overlap = []
    for coarse_id, fine_id in coarse_to_fine.items():
        coarse_graph = coarse_lookup.get(coarse_id)
        fine_graph = fine_lookup.get(fine_id)
        if coarse_graph is None or fine_graph is None:
            print(f" Skipping coarse {coarse_id}", flush=True)
            continue
        print(f"Interpolating coarse {coarse_id} to fine {fine_id}",
              flush=True)
        g = copy.deepcopy(coarse_graph)
        g.y_coeffs = fine_graph.y_coeffs

        coarse_pos = coarse_graph.pos.numpy()
        fine_pos = fine_graph.pos.numpy()
        fine_vel = fine_graph.y_velocity.numpy()
        fine_pres = fine_graph.y_pressure.numpy().flatten()
        coarse_vel = coarse_graph.y_velocity.numpy()
        coarse_pres = coarse_graph.y_pressure.numpy().flatten()
        wall_mask = coarse_graph.node_types.numpy() == 1

        vel = interpolate_fine_to_coarse(
            fine_centers=fine_pos,
            fine_values=fine_vel,
            coarse_centers=coarse_pos,
            coarse_values=coarse_vel,
            wall_mask=wall_mask
        )
        pres = interpolate_fine_to_coarse(
            fine_centers=fine_pos,
            fine_values=fine_pres[:, np.newaxis],
            coarse_centers=coarse_pos,
            coarse_values=coarse_pres[:, np.newaxis],
            wall_mask=wall_mask
        )
        import torch
        g.y_velocity = torch.tensor(vel, dtype=torch.float32)
        g.y_pressure = torch.tensor(
            pres.flatten(), dtype=torch.float32).unsqueeze(1)

        fine_overlap.append(g)

    return fine_overlap


def cv_mf():
    
    set_seed(42)
    dataset = MultiFidelityDataset()
    coarse_data = list(dataset.coarse)
    fine_data = list(dataset.fine)

    coarse_to_fine = get_coarse_to_fine()
    fine_lookup = {d.id: d for d in fine_data}
    coarse_lookup = {d.id: d for d in coarse_data}
    fine_overlap = build_fine_overlap(coarse_lookup, fine_lookup, coarse_to_fine)

    results = cross_validation_mf(coarse_data, fine_overlap, coarse_epochs=500, correction_epochs=500)

    print_results(results)
    return results


if __name__ == "__main__":
    cv_mf()    