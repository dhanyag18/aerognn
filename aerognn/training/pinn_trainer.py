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
                 lambda_data=1.0, lambda_cont=0.05,
                 lambda_mom=0.001):
        self.model = model
        self.physics = physics_loss
        self.optimizer = optimizer
        self.lambdas = {
            'data': lambda_data,
            'continuity': lambda_cont,
            'momentum': lambda_mom,
        }
        self.loss_history = {
            'total': [],
            'data': [],
            'continuity': [],
            'momentum': [],
        }

    def train_step(self, batch, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(batch)
        L_data = self._data_loss(output, batch)
        interior_mask = (batch.node_types == 0)

        L_cont = torch.tensor(0.0)
        L_mom = torch.tensor(0.0)

        if epoch < 100:
            L_total = L_data
        elif epoch < 200:
            L_cont = self.physics.continuity_residual(
                output['velocity'], batch.pos, batch.edge_index, interior_mask)
            if torch.isfinite(L_cont):
                L_total = L_data + self.lambdas['continuity'] * L_cont
            else:
                L_cont = torch.tensor(0.0)
                L_total = L_data
        else:
            L_cont = self.physics.continuity_residual(
                output['velocity'], batch.pos, batch.edge_index, interior_mask)
            L_mom = self.physics.momentum_residual(
                output['velocity'], output['pressure'],
                batch.pos, batch.edge_index, interior_mask)
            L_cont = L_cont if torch.isfinite(L_cont) else torch.tensor(0.0)
            L_mom = L_mom if torch.isfinite(L_mom) else torch.tensor(0.0)
            L_total = (L_data
                      + self.lambdas['continuity'] * L_cont
                      + self.lambdas['momentum'] * L_mom)

        L_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        self.loss_history['total'].append(L_total.item())
        self.loss_history['data'].append(L_data.item())
        self.loss_history['continuity'].append(L_cont.item())
        self.loss_history['momentum'].append(L_mom.item())

    def _data_loss(self, output, batch):
        criterion = torch.nn.MSELoss()
        L_velocity = criterion(output['velocity'], batch.y_velocity)
        L_pressure = criterion(output['pressure'].squeeze(), batch.y_pressure.squeeze())
        L_cd = criterion(output['scores'][:, 0], batch.y_coeffs[:, 0])
        L_cl = criterion(output['scores'][:, 1], batch.y_coeffs[:, 1])
        L_clstd = criterion(output['scores'][:, 2], batch.y_coeffs[:, 2])
        L_coeffs = L_cd + 5 * L_cl + 100 * L_clstd
        return 0.005 * L_velocity + 0.0001 * L_pressure + 1 * L_coeffs

    def _compute_score(self, output):
        cd = output['scores'][:, 0]
        cl = output['scores'][:, 1]
        cl_std = output['scores'][:, 2]
        return 0.2 * cd + 0.2 * torch.abs(cl) + 0.6 * cl_std

    def evaluate(self, loader):
        self.model.eval()
        score_preds, score_actuals = [], []
        cd_preds, cd_actuals = [], []
        cl_preds, cl_actuals = [], []
        clstd_preds, clstd_actuals = [], []
        vel_preds, vel_actuals = [], []
        pres_preds, pres_actuals = [], []
        criterion = torch.nn.MSELoss()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                output = self.model(batch)
                L_v = criterion(output['velocity'], batch.y_velocity)
                L_p = criterion(output['pressure'].squeeze(), batch.y_pressure.squeeze())
                L_cd = criterion(output['scores'][:, 0], batch.y_coeffs[:, 0])
                L_cl = criterion(output['scores'][:, 1], batch.y_coeffs[:, 1])
                L_clstd = criterion(output['scores'][:, 2], batch.y_coeffs[:, 2])
                L_c = L_cd + 5 * L_cl + 100 * L_clstd
                total_loss += (0.005*L_v + 0.0001*L_p + 1*L_c).item() * batch.num_graphs

                pred_score = self._compute_score(output)
                cd = batch.y_coeffs[:, 0]
                cl = batch.y_coeffs[:, 1]
                cl_std = batch.y_coeffs[:, 2]
                true_score = 0.2 * cd + 0.2 * torch.abs(cl) + 0.6 * cl_std
                score_preds.extend(pred_score.tolist())
                score_actuals.extend(true_score.tolist())

                cd_preds.extend(output['scores'][:, 0].tolist())
                cd_actuals.extend(batch.y_coeffs[:, 0].tolist())
                cl_preds.extend(output['scores'][:, 1].tolist())
                cl_actuals.extend(batch.y_coeffs[:, 1].tolist())
                clstd_preds.extend(output['scores'][:, 2].tolist())
                clstd_actuals.extend(batch.y_coeffs[:, 2].tolist())

                vel_preds.extend(output['velocity'].tolist())
                vel_actuals.extend(batch.y_velocity.tolist())
                pres_preds.extend(output['pressure'].squeeze().tolist())
                pres_actuals.extend(batch.y_pressure.squeeze().tolist())

        vel_preds = np.array(vel_preds)
        vel_actuals = np.array(vel_actuals)

        return {
            'loss': total_loss/len(loader.dataset),
            'r2_score': r2_score(score_actuals, score_preds),
            'mae_score': np.mean(np.abs(np.array(score_preds) - np.array(score_actuals))),
            'r2_cd': r2_score(cd_actuals, cd_preds),
            'mae_cd': np.mean(np.abs(np.array(cd_preds) - np.array(cd_actuals))),
            'r2_cl': r2_score(cl_actuals, cl_preds),
            'mae_cl': np.mean(np.abs(np.array(cl_preds) - np.array(cl_actuals))),
            'r2_clstd': r2_score(clstd_actuals, clstd_preds),
            'mae_clstd': np.mean(np.abs(np.array(clstd_preds) - np.array(clstd_actuals))),
            'r2_vx': r2_score(vel_actuals[:, 0], vel_preds[:, 0]),
            'mae_vx': np.mean(np.abs(vel_actuals[:, 0] - vel_preds[:, 0])),
            'r2_vy': r2_score(vel_actuals[:, 1], vel_preds[:, 1]),
            'mae_vy': np.mean(np.abs(vel_actuals[:, 1] - vel_preds[:, 1])),
            'r2_vz': r2_score(vel_actuals[:, 2], vel_preds[:, 2]),
            'mae_vz': np.mean(np.abs(vel_actuals[:, 2] - vel_preds[:, 2])),
            'r2_pres': r2_score(pres_actuals, pres_preds),
            'mae_pres': np.mean(np.abs(np.array(pres_preds) - np.array(pres_actuals))),
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

    fold_metrics = {k: [] for k in [
        'score_mae', 'score_r2',
        'cd_mae', 'cd_r2',
        'cl_mae', 'cl_r2',
        'clstd_mae', 'clstd_r2',
        'vx_mae', 'vx_r2',
        'vy_mae', 'vy_r2',
        'vz_mae', 'vz_r2',
        'pres_mae', 'pres_r2',
    ]}

    for fold, (train_idx, test_idx) in enumerate(cv_strategy.split(range(len(dataset)), groups=groups)):
        train_x = DataLoader([dataset[i] for i in train_idx], batch_size=8, shuffle=True)
        test_x = DataLoader([dataset[i] for i in test_idx], batch_size=8, shuffle=False)

        model = PINNGraphNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5, min_lr=1e-5)
        trainer = PINNTrainer(model, PhysicsLoss(), optimizer)

        best_val_loss = float('inf')
        best_model_state = None
        patience = 50
        no_improve_count = 0
        curriculum_transitions = {99, 199}

        for epoch in range(epochs):
            for batch in train_x:
                trainer.train_step(batch, epoch)
            
            metrics = trainer.evaluate(test_x)
            val_loss = metrics['loss']
            scheduler.step(val_loss)

            if epoch in curriculum_transitions:
                best_val_loss = float('inf')
                no_improve_count = 0
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=20, factor=0.5, min_lr=1e-5)
                trainer.optimizer = optimizer
                print(f"Epoch {epoch+1}: New physics loss introduced, resetting early stopping", flush=True)
                continue

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

        model.load_state_dict(best_model_state)
        metrics = trainer.evaluate(test_x)
        

        fold_metrics['score_mae'].append(metrics['mae_score'])
        fold_metrics['score_r2'].append(metrics['r2_score'])
        fold_metrics['cd_mae'].append(metrics['mae_cd'])
        fold_metrics['cd_r2'].append(metrics['r2_cd'])
        fold_metrics['cl_mae'].append(metrics['mae_cl'])
        fold_metrics['cl_r2'].append(metrics['r2_cl'])
        fold_metrics['clstd_mae'].append(metrics['mae_clstd'])
        fold_metrics['clstd_r2'].append(metrics['r2_clstd'])
        fold_metrics['vx_mae'].append(metrics['mae_vx'])
        fold_metrics['vx_r2'].append(metrics['r2_vx'])
        fold_metrics['vy_mae'].append(metrics['mae_vy'])
        fold_metrics['vy_r2'].append(metrics['r2_vy'])
        fold_metrics['vz_mae'].append(metrics['mae_vz'])
        fold_metrics['vz_r2'].append(metrics['r2_vz'])
        fold_metrics['pres_mae'].append(metrics['mae_pres'])
        fold_metrics['pres_r2'].append(metrics['r2_pres'])

    return {f'avg_{k}': np.mean(v) for k, v in fold_metrics.items()}