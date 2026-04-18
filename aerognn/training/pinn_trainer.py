import torch
import numpy as np
import copy
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from sklearn.model_selection import GroupKFold
from aerognn.training.physics_loss import PhysicsLoss
from aerognn.models.pinn_graphnet import PINNGraphNet

class PINNTrainer:

    def __init__(self, model, physics_loss, optimizer,
                 lambda_data=1.0, lambda_cont=0.1,
                 lambda_mom=0.01, lambda_bc=1.0):
        self.model = model
        self.physics = physics_loss
        self.optimizer = optimizer
        self.lambdas = {
            'data': lambda_data,
            'continuity': lambda_cont,
            'momentum': lambda_mom,
            'boundary': lambda_bc,
        }
        self.loss_history = {k: [] for k in self.lambdas}
        
    def train_step(self, batch, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(batch)
        L_data = self._data_loss(output, batch)
        interior_mask = (batch.node_types == 0)
        L_bc = self.physics.boundary_loss(
            output['velocity'], batch.node_types, batch.abl_velocity)

        if epoch < 50:
            L_total = L_data + L_bc

        elif epoch < 150:
            L_cont = self.physics.continuity_residual(
                output['velocity'], batch.pos, batch.edge_index, interior_mask)
            L_total = L_data + L_bc + 0.01 * L_cont

        else:
            L_cont = self.physics.continuity_residual(
                output['velocity'], batch.pos, batch.edge_index, interior_mask)
            L_mom = self.physics.momentum_residual(
                output['velocity'], output['pressure'],
                batch.pos, batch.edge_index, interior_mask)
            L_total = L_data + L_bc + 0.1 * L_cont + 0.01 * L_mom

        L_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        losses = {'total': L_total.item(), 'data': L_data.item(), 'boundary': L_bc.item()}
        for k, v in losses.items():
            if k in self.loss_history:
                self.loss_history[k].append(v)

    def _data_loss(self, output, batch):
        criterion = torch.nn.MSELoss()
        L_velocity = criterion(output['velocity'], batch.y_velocity)
        L_pressure = criterion(output['pressure'].squeeze(), batch.y_pressure.squeeze())
        L_scores = criterion(output['scores'], batch.y)
        return L_velocity + L_pressure + L_scores

    def compute_final_score(self, output):
        cl_std = output['scores'][:, 0]
        cl_mean = output['scores'][:, 1]
        cd_mean = output['scores'][:, 2]
        return 0.6*cl_std + 0.2*cl_mean + 0.2*cd_mean

    def evaluate(self, loader):
        self.model.eval()
        final_preds, final_actuals = [], []
        vel_preds, vel_actuals = [], []
        pres_preds, pres_actuals = [], []
        criterion = torch.nn.MSELoss()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                output = self.model(batch)
                L_v = criterion(output['velocity'], batch.y_velocity)
                L_p = criterion(output['pressure'].squeeze(), batch.y_pressure.squeeze())
                L_s = criterion(output['scores'], batch.y)
                total_loss += (L_v + L_p + L_s).item() * batch.num_graphs
                pred_score = self.compute_final_score(output)
                true_score = (0.6*batch.y[:, 0] + 0.2*batch.y[:, 1] + 0.2*batch.y[:, 2])
                final_preds.extend(pred_score.tolist())
                final_actuals.extend(true_score.tolist())
                vel_preds.extend(output['velocity'].tolist())
                vel_actuals.extend(batch.y_velocity.tolist())
                pres_preds.extend(output['pressure'].squeeze().tolist())
                pres_actuals.extend(batch.y_pressure.squeeze().tolist())

        vel_preds = np.array(vel_preds)
        vel_actuals = np.array(vel_actuals)
        pres_preds = np.array(pres_preds)
        pres_actuals = np.array(pres_actuals)

        return {
            'loss': total_loss/len(loader.dataset),
            'r2_scores': r2_score(final_actuals, final_preds),
            'mae_scores': sum(abs(p-a) for p,a in zip(final_preds, final_actuals))/len(final_preds),
            'r2_vx': r2_score(vel_actuals[:, 0], vel_preds[:, 0]),
            'mae_vx': np.mean(np.abs(vel_actuals[:, 0] - vel_preds[:, 0])),
            'r2_vy': r2_score(vel_actuals[:, 1], vel_preds[:, 1]),
            'mae_vy': np.mean(np.abs(vel_actuals[:, 1] - vel_preds[:, 1])),
            'r2_vz': r2_score(vel_actuals[:, 2], vel_preds[:, 2]),
            'mae_vz': np.mean(np.abs(vel_actuals[:, 2] - vel_preds[:, 2])),
            'r2_pres': r2_score(pres_actuals, pres_preds),
            'mae_pres': np.mean(np.abs(pres_actuals - pres_preds)),
        }


def cross_validation_pinn(dataset, epochs):
    
    BATCH_GROUPS = {
        **{i: f"rand_{i}" for i in range(1, 72)},
        **{i: f"rand_{i}" for i in range(105, 126)},
        **{i: "explore_setback" for i in range(77, 82)},
        **{i: "explore_m" for i in range(82, 87)},
        **{i: "explore_bulge" for i in range(87, 91)},
        **{i: "explore_chamfer" for i in range(91, 95)},
        **{i: f"explore_{i}" for i in [72, 73, 74, 75, 76] + list(range(95, 105))},
        **{i: "old_gp_grid" for i in range(126, 135)},
        **{i: "old_xgb_grid" for i in range(135, 145)},
        **{i: "grid_batch_1" for i in range(145, 155)},
        **{i: "grid_batch_2" for i in range(155, 165)},
        **{i: "grid_batch_3" for i in range(165, 175)},
        **{i: "grid_batch_4" for i in range(175, 190)},
        **{i: "grid_batch_5" for i in range(190, 200)},
        **{i: "grid_batch_6" for i in range(200, 210)},
        **{i: f"de_gp_{i}" for i in range(210, 215)},
        **{i: f"de_xgb_{i}" for i in range(215, 220)},
        **{i: f"val_{i}" for i in range(220, 230)},
        **{i: f"xgb_opt_{i}" for i in range(230, 235)},
        **{i: f"gp_opt_{i}" for i in range(235, 240)},
        **{i: f"batch7_{i}" for i in range(240, 260)},
        **{i: f"batch8_{i}" for i in range(260, 295)},
        **{i: f"diverse_exploration_{i}" for i in range(295, 306)},
        **{i: f"diverse_exploration_2{i}" for i in range(306, 323)},
        **{i: f"diverse_exploration_3{i}" for i in range(323, 439)},
        **{i: f"optimized{i}" for i in range(439, 454)},
        **{i: f"optimized_2{i}" for i in range(454, 469)},
        **{i: f"optimized_3{i}" for i in range(469, 484)},
        **{i: f"optimized_4{i}" for i in range(484, 499)}
    }

    groups = [BATCH_GROUPS[d.id] for d in dataset]
    cv_strategy = GroupKFold(n_splits=10)

    fold_scores_mae = []
    fold_scores_r2 = []
    fold_vx_mae = []
    fold_vx_r2 = []
    fold_vy_mae = []
    fold_vy_r2 = []
    fold_vz_mae = []
    fold_vz_r2 = []
    fold_pres_mae = []
    fold_pres_r2 = []

    for train_idx, test_idx in cv_strategy.split(range(len(dataset)), groups=groups):
        train_x = DataLoader([dataset[i] for i in train_idx], batch_size=8, shuffle=False)
        test_x = DataLoader([dataset[i] for i in test_idx], batch_size=8, shuffle=False)

        model = PINNGraphNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5)
        trainer = PINNTrainer(model, PhysicsLoss(), optimizer)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = 30
        no_improve_count = 0

        for epoch in range(epochs):
            for batch in train_x:
                trainer.train_step(batch, epoch)

            metrics = trainer.evaluate(test_x)
            val_loss = metrics['loss']
            scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}, Loss: {val_loss:.4f}, MAE: {metrics['mae_scores']:.4f}, R2: {metrics['r2_scores']:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(best_model_state)
        metrics = trainer.evaluate(test_x)

        fold_scores_mae.append(metrics['mae_scores'])
        fold_scores_r2.append(metrics['r2_scores'])
        fold_vx_mae.append(metrics['mae_vx'])
        fold_vx_r2.append(metrics['r2_vx'])
        fold_vy_mae.append(metrics['mae_vy'])
        fold_vy_r2.append(metrics['r2_vy'])
        fold_vz_mae.append(metrics['mae_vz'])
        fold_vz_r2.append(metrics['r2_vz'])
        fold_pres_mae.append(metrics['mae_pres'])
        fold_pres_r2.append(metrics['r2_pres'])

    return {
        'avg_scores_mae': sum(fold_scores_mae)/len(fold_scores_mae),
        'avg_scores_r2': sum(fold_scores_r2)/len(fold_scores_r2),
        'avg_vx_mae': sum(fold_vx_mae)/len(fold_vx_mae),
        'avg_vx_r2': sum(fold_vx_r2)/len(fold_vx_r2),
        'avg_vy_mae': sum(fold_vy_mae)/len(fold_vy_mae),
        'avg_vy_r2': sum(fold_vy_r2)/len(fold_vy_r2),
        'avg_vz_mae': sum(fold_vz_mae)/len(fold_vz_mae),
        'avg_vz_r2': sum(fold_vz_r2)/len(fold_vz_r2),
        'avg_pres_mae': sum(fold_pres_mae)/len(fold_pres_mae),
        'avg_pres_r2': sum(fold_pres_r2)/len(fold_pres_r2),
    }