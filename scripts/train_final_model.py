import torch
import numpy as np
import copy
from torch.utils.data import random_split
from aerognn.data.dataset import BuildingDataset
from aerognn.data.multi_fidelity_dataset import MultiFidelityDataset
from aerognn.models.gcn_surrogate import BuildingGCN
from aerognn.models.pinn_graphnet import PINNGraphNet
from aerognn.models.mutli_fidelity import MultiFidelityPINNGNN
from aerognn.training.pinn_trainer import PINNTrainer, PhysicsLoss
from aerognn.training.mf_trainer import MFTrainer
from torch_geometric.loader import DataLoader
from aerognn.training.trainer import train_epoch, evaluate
from aerognn.data.interpolate import interpolate_fine_to_coarse
from aerognn.data.groups import get_coarse_to_fine
import random


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def interpolate_graph(coarse_graph, fine_graph):
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

    velocity_tensor = torch.tensor(vel, dtype=torch.float32)
    pressure_tensor = torch.tensor(
        pres.flatten(), dtype=torch.float32).unsqueeze(1)

    return velocity_tensor, pressure_tensor


def train_final_model():
    dataset = BuildingDataset()
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_x, test_x = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_x, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_x, batch_size=32, shuffle=False)

    final_model = BuildingGCN()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5)

    best_val_loss = float('inf')
    best_model_state = None
    patience = 30
    no_improve_count = 0

    for i in range(300):
        loss, mae = train_epoch(final_model, train_loader, optimizer, criterion)
        val_loss, val_mae, val_r2 = evaluate(final_model, test_loader, criterion)
        scheduler.step(val_loss)
        if (i + 1) % 20 == 0:
            print(f"Epoch {i+1}, Loss: {loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(final_model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            break

    final_model.load_state_dict(best_model_state)
    torch.save(final_model.state_dict(), "final_model.pt")
    print("Model saved to final_model.pt")


def train_final_pinn_model():
    set_seed(42)
    dataset = BuildingDataset()
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_x, test_x = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_x, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_x, batch_size=8, shuffle=False)

    final_model = PINNGraphNet()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-5)
    trainer = PINNTrainer(final_model, PhysicsLoss(), optimizer)

    best_val_loss = float('inf')
    best_model_state = None
    patience = 50
    no_improve_count = 0
    curriculum_transitions = {99, 199}

    for epoch in range(400):
        for batch in train_loader:
            trainer.train_step(batch, epoch)
        metrics = trainer.evaluate(test_loader)
        val_loss = metrics['loss']
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {val_loss:.4f}")

        if epoch in curriculum_transitions:
            best_val_loss = float('inf')
            no_improve_count = 0
            optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=20, factor=0.5, min_lr=1e-5)
            trainer.optimizer = optimizer
            print(f"Epoch {epoch+1}: New physics loss introduced", flush=True)
            continue

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(final_model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}", flush=True)
            break

    final_model.load_state_dict(best_model_state)
    torch.save(final_model.state_dict(), "final_pinn_model.pt")
    print("Model saved to final_pinn_model.pt")


def train_mf():
    set_seed(42)
    dataset = MultiFidelityDataset()

    coarse_data = list(dataset.coarse)
    fine_data   = list(dataset.fine)

    coarse_to_fine = get_coarse_to_fine()

    fine_lookup   = {d.id: d for d in fine_data}
    coarse_lookup = {d.id: d for d in coarse_data}

    fine_overlap = []
    for coarse_id, fine_id in coarse_to_fine.items():
        coarse_graph = coarse_lookup.get(coarse_id)
        fine_graph   = fine_lookup.get(fine_id)
        if coarse_graph is None or fine_graph is None:
            print(f"  Skipping coarse {coarse_id} -> fine {fine_id} (not found)")
            continue
        print(f"  Interpolating coarse {coarse_id} -> fine {fine_id}...",
              flush=True)
        g = copy.deepcopy(coarse_graph)
        g.y_coeffs = fine_graph.y_coeffs
        vel, pres = interpolate_graph(coarse_graph, fine_graph)
        g.y_velocity = vel
        g.y_pressure = pres
        fine_overlap.append(g)

    print(f"Built {len(fine_overlap)} coarse graphs with fine targets")
    print(f"Training on {len(coarse_data)} coarse simulations")

    train_loader = DataLoader(coarse_data, batch_size=8, shuffle=True)

    random.seed(42)
    fine_overlap_shuffled = fine_overlap.copy()
    random.shuffle(fine_overlap_shuffled)
    n_fine = len(fine_overlap_shuffled)
    n_fine_train = int(0.8 * n_fine)
    fine_train = DataLoader(
        fine_overlap_shuffled[:n_fine_train], batch_size=4, shuffle=True)
    fine_test = DataLoader(
        fine_overlap_shuffled[n_fine_train:], batch_size=4, shuffle=False)

    print(f"Fine train: {n_fine_train} sims, Fine test: {n_fine - n_fine_train} sims")

    mf_model = MultiFidelityPINNGNN()
    optimizer = torch.optim.Adam(mf_model.coarse_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-5)
    trainer = MFTrainer(mf_model, PhysicsLoss(), optimizer)

    best_val_loss = float('inf')
    best_coarse_state = None
    patience = 50
    no_improve_count = 0
    curriculum_transitions = {99, 199}

    print("Stage 1: Coarse training...")
    for epoch in range(500):
        for batch in train_loader:
            trainer.train_step(batch, epoch)

        metrics = trainer.evaluate_coarse(fine_test)
        val_loss = metrics['loss']
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Coarse Epoch {epoch+1} | Loss: {val_loss:.4f} | "
                  f"Score R2: {metrics['r2_score']:.4f} | "
                  f"Cd R2: {metrics['r2_cd']:.4f}", flush=True)

        if epoch in curriculum_transitions:
            best_val_loss = float('inf')
            no_improve_count = 0
            optimizer = torch.optim.Adam(
                mf_model.coarse_model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=20, factor=0.5, min_lr=1e-5)
            trainer.optimizer = optimizer
            print(f"Epoch {epoch+1}: New physics loss introduced", flush=True)
            continue

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_coarse_state = copy.deepcopy(mf_model.coarse_model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Coarse early stopping at epoch {epoch+1}", flush=True)
            break

    mf_model.coarse_model.load_state_dict(best_coarse_state)

    for param in mf_model.coarse_model.parameters():
        param.requires_grad = False

    correction_optimizer = torch.optim.Adam(
        mf_model.correction_model.parameters(), lr=0.001)
    correction_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        correction_optimizer, patience=10, factor=0.5, min_lr=1e-5)
    correction_trainer = MFTrainer(mf_model, PhysicsLoss(), correction_optimizer)

    best_correction_loss = float('inf')
    best_mf_state = None
    correction_patience = 100
    correction_no_improve = 0

    print("Stage 2: Correction training...")
    for epoch in range(500):
        for batch in fine_train:
            correction_trainer.correction_train_step(batch)

        metrics = correction_trainer.evaluate(fine_test)
        val_loss = metrics['loss']
        correction_scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Correction Epoch {epoch+1} | Loss: {val_loss:.4f} | "
                  f"Score R2: {metrics['r2_score']:.4f} | "
                  f"Cd R2: {metrics['r2_cd']:.4f}", flush=True)

        if val_loss < best_correction_loss:
            best_correction_loss = val_loss
            best_mf_state = copy.deepcopy(mf_model.state_dict())
            correction_no_improve = 0
        else:
            correction_no_improve += 1
        if correction_no_improve >= correction_patience:
            print(f"Correction early stopping at epoch {epoch+1}", flush=True)
            break

    mf_model.load_state_dict(best_mf_state)
    torch.save(mf_model.state_dict(), "final_mf_model.pt")
    print("Model saved to final_mf_model.pt")


if __name__ == "__main__":
    train_mf()