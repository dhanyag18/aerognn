import torch
import copy
import os
import numpy as np
from aerognn.data.multi_fidelity_dataset import MultiFidelityDataset
from aerognn.data.interpolate import interpolate_fine_to_coarse
from aerognn.models.multi_fidelity import MultiFidelityPINNGNN
from aerognn.simulation.case_generator import generate_case
from aerognn.simulation.runner import SimulationRunner
from aerognn.simulation.result_extractor import extract_simulation_result
from aerognn.active_learning.acquisition import AcquisitionFunction
from aerognn.active_learning.controller import ActiveLearningController
from aerognn.training.mf_trainer import MFRetrainer, MFTrainer
from aerognn.training.physics_loss import PhysicsLoss
from aerognn.data.groups import get_coarse_to_fine
from torch_geometric.loader import DataLoader
import csv

HOST_AEROGNN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(os.path.dirname(HOST_AEROGNN), 'openfoam', 'vortex')
OUTPUT_DIR   = os.path.join(HOST_AEROGNN, 'active_learning_cases')


COARSE_TO_FINE = get_coarse_to_fine()


def build_fine_overlap(coarse_lookup, fine_lookup):
    fine_overlap = []
    for coarse_id, fine_id in COARSE_TO_FINE.items():
        coarse_graph = coarse_lookup.get(coarse_id)
        fine_graph = fine_lookup.get(fine_id)
        if coarse_graph is None or fine_graph is None:
            continue
        
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

        g.y_velocity = torch.tensor(vel, dtype=torch.float32)
        g.y_pressure = torch.tensor(pres.flatten(), dtype=torch.float32).unsqueeze(1)
                
        fine_overlap.append(g)
    return fine_overlap


def evaluate_model(model, fine_overlap):
    fine_test = DataLoader(fine_overlap, batch_size=4, shuffle=False)
    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = MFTrainer(model, PhysicsLoss(), dummy_optimizer)
    return trainer.evaluate(fine_test)


def run_active_learning():
    model = MultiFidelityPINNGNN()
    model.load_state_dict(torch.load(os.path.join(HOST_AEROGNN, 'final_mf_model.pt')))
    model.eval()

    dataset = MultiFidelityDataset()
    coarse_data = list(dataset.coarse)
    fine_data = list(dataset.fine)

    coarse_lookup = {d.id: d for d in coarse_data}
    fine_lookup = {d.id: d for d in fine_data}

    fine_overlap = build_fine_overlap(coarse_lookup, fine_lookup)

    physics = PhysicsLoss()
    runner = SimulationRunner(
        container_name='openfoam_daemon',
        host_base_path=HOST_AEROGNN,
        container_base_path='/home/openfoam'
    )
    retrainer = MFRetrainer(fine_overlap, coarse_epochs=500)
    acquisition = AcquisitionFunction(model, physics, alpha=0.5, beta=0.1)

    controller = ActiveLearningController(
        model=model,
        generate_case_fn=generate_case,
        template_dir=TEMPLATE_DIR,
        output_dir=OUTPUT_DIR,
        runner=runner,
        extract_fn=extract_simulation_result,
        dataset=dataset,
        trainer=retrainer,
        acquisition=acquisition
    )

    metrics = evaluate_model(model, fine_overlap)
    iteration_metrics = [{
        'iteration': 0,
        'n_coarse': len(list(dataset.coarse)),
        'score_r2': metrics['r2_score'],
        'cd_r2': metrics['r2_cd'],
        'cl_r2': metrics['r2_cl'],
        'clstd_r2': metrics['r2_clstd'],
    }]

    n_iterations = 2
    for i in range(n_iterations):
        n_success = controller.run_iteration(
            n_designs=5,
            resolution='coarse',
            n_random_candidates=100,
            iteration=i + 1
        )
        if n_success == 0:
            print('No successful simulations.')
            break

        metrics = evaluate_model(model, fine_overlap)
        iteration_metrics.append({
            'iteration': i + 1,
            'n_coarse': len(list(dataset.coarse)),
            'score_r2': metrics['r2_score'],
            'cd_r2': metrics['r2_cd'],
            'cl_r2': metrics['r2_cl'],
            'clstd_r2': metrics['r2_clstd'],
        })

    metrics_path = 'data/coarse/iteration_metrics.csv'
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'iteration', 'n_coarse', 'score_r2',
            'cd_r2', 'cl_r2', 'clstd_r2'])
        writer.writeheader()
        writer.writerows(iteration_metrics)
    print(f"\nMetrics saved to {metrics_path}")

    return iteration_metrics

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_active_learning()