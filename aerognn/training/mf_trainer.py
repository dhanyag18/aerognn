import torch
import numpy as np
import copy
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from sklearn.model_selection import GroupKFold
from aerognn.training.physics_loss import PhysicsLoss
from aerognn.models.mutli_fidelity import MultiFidelityPINNGNN
from aerognn.data.groups import get_coarse_groups, get_groups

class MFTrainer:

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

    def _data_loss(self, output, batch):
        criterion = torch.nn.MSELoss()
        L_velocity = criterion(output['velocity'], batch.y_velocity)
        L_pressure = criterion(output['pressure'].squeeze(), batch.y_pressure.squeeze())
        L_cd = criterion(output['scores'][:, 0], batch.y_coeffs[:, 0])
        L_cl = criterion(output['scores'][:, 1], batch.y_coeffs[:, 1])
        L_clstd = criterion(output['scores'][:, 2], batch.y_coeffs[:, 2])
        L_coeffs = L_cd + 5 * L_cl + 100 * L_clstd
        return 0.005 * L_velocity + 0.0001 * L_pressure + 1 * L_coeffs

    def train_step(self, batch, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model.coarse_model(batch)
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
        torch.nn.utils.clip_grad_norm_(self.model.coarse_model.parameters(), 1)
        self.optimizer.step()

        self.loss_history['total'].append(L_total.item())
        self.loss_history['data'].append(L_data.item())
        self.loss_history['continuity'].append(L_cont.item())
        self.loss_history['momentum'].append(L_mom.item())

    def evaluate_coarse(self, loader):
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
                output = self.model.coarse_model(batch)
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

        return self._compute_metrics(
            total_loss, loader, score_preds, score_actuals,
            cd_preds, cd_actuals, cl_preds, cl_actuals,
            clstd_preds, clstd_actuals, vel_preds, vel_actuals,
            pres_preds, pres_actuals)

    def correction_train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(batch)
        L_data = self._data_loss(output, batch)
        L_data.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.correction_model.parameters(), 1)
        self.optimizer.step()
        self.loss_history['total'].append(L_data.item())
        self.loss_history['data'].append(L_data.item())

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

        return self._compute_metrics(
            total_loss, loader, score_preds, score_actuals,
            cd_preds, cd_actuals, cl_preds, cl_actuals,
            clstd_preds, clstd_actuals, vel_preds, vel_actuals,
            pres_preds, pres_actuals)

    def _compute_metrics(self, total_loss, loader, score_preds, score_actuals,
                         cd_preds, cd_actuals, cl_preds, cl_actuals,
                         clstd_preds, clstd_actuals, vel_preds, vel_actuals,
                         pres_preds, pres_actuals):
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

    def _compute_score(self, output):
        cd = output['scores'][:, 0]
        cl = output['scores'][:, 1]
        cl_std = output['scores'][:, 2]
        return 0.2 * cd + 0.2 * torch.abs(cl) + 0.6 * cl_std


class MFRetrainer:
    def __init__(self, fine_overlap, coarse_epochs=500, correction_epochs=500):
        self.fine_overlap = fine_overlap
        self.coarse_epochs = coarse_epochs
        self.correction_epochs = correction_epochs

    def retrain(self, model, dataset):
        coarse_data = list(dataset.coarse)
        train_loader = DataLoader(coarse_data, batch_size=8, shuffle=True)

        optimizer = torch.optim.Adam(
            model.coarse_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5, min_lr=1e-5)
        trainer = MFTrainer(model, PhysicsLoss(), optimizer)

        best_val_loss = float('inf')
        best_state = None
        patience = 50
        no_improve = 0
        curriculum_transitions = {99, 199}

        fine_test = DataLoader(
            self.fine_overlap, batch_size=4, shuffle=False)

        print(f"Retraining coarse model on {len(coarse_data)} simulations")
        for epoch in range(self.coarse_epochs):
            for batch in train_loader:
                trainer.train_step(batch, epoch)

            metrics = trainer.evaluate_coarse(fine_test)
            val_loss = metrics['loss']
            scheduler.step(val_loss)

            if epoch in curriculum_transitions:
                best_val_loss = float('inf')
                no_improve = 0
                optimizer = torch.optim.Adam(
                    model.coarse_model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=20, factor=0.5, min_lr=1e-5)
                trainer.optimizer = optimizer
                continue

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.coarse_model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        model.coarse_model.load_state_dict(best_state)
        print("Retraining complete.")


def cross_validation_mf(coarse_dataset, fine_dataset, coarse_epochs, correction_epochs):

    FINE_GROUPS = get_groups()
    fine_labels = [FINE_GROUPS[g.id] for g in fine_dataset]
    fine_cv = GroupKFold(n_splits=5, shuffle = True, random_state = 42)
    fine_folds = list(fine_cv.split(range(len(fine_dataset)), groups=fine_labels))

    print(f"Fine dataset: {len(fine_dataset)} paired graphs, 5-fold GroupKFold CV")

    COARSE_GROUPS = get_coarse_groups()
    coarse_labels = [COARSE_GROUPS[d.id] for d in coarse_dataset]
    coarse_cv = GroupKFold(n_splits=10)

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

    for fold, (train_idx, test_idx) in enumerate(
            coarse_cv.split(range(len(coarse_dataset)), groups=coarse_labels)):

        fine_fold_idx = fold % 5
        fine_train_idx, fine_test_idx = fine_folds[fine_fold_idx]
        fine_train = DataLoader([fine_dataset[i] for i in fine_train_idx], batch_size=4, shuffle=True)
        fine_test = DataLoader([fine_dataset[i] for i in fine_test_idx], batch_size=4, shuffle=False)

        train_x = DataLoader([coarse_dataset[i] for i in train_idx], batch_size=8, shuffle=True)
        test_x = DataLoader([coarse_dataset[i] for i in test_idx], batch_size=8, shuffle=False)

        mf_model = MultiFidelityPINNGNN()
        optimizer = torch.optim.Adam(mf_model.coarse_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-5)
        trainer = MFTrainer(mf_model, PhysicsLoss(), optimizer)

        best_val_loss = float('inf')
        best_coarse_state = None
        patience = 50
        no_improve_count = 0
        curriculum_transitions = {99, 199}

        for epoch in range(coarse_epochs):
            for batch in train_x:
                trainer.train_step(batch, epoch)

            metrics = trainer.evaluate_coarse(test_x)
            val_loss = metrics['loss']
            scheduler.step(val_loss)

            if epoch in curriculum_transitions:
                best_val_loss = float('inf')
                no_improve_count = 0
                optimizer = torch.optim.Adam(
                    mf_model.coarse_model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, min_lr=1e-5)
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

        correction_optimizer = torch.optim.Adam(mf_model.correction_model.parameters(), lr=0.0005)
        correction_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(correction_optimizer, patience=20, factor=0.5, min_lr=1e-6)
        correction_trainer = MFTrainer(mf_model, PhysicsLoss(), correction_optimizer)

        best_correction_loss = float('inf')
        best_mf_state = None
        correction_patience = 100
        correction_no_improve = 0

        for epoch in range(correction_epochs):
            for batch in fine_train:
                correction_trainer.correction_train_step(batch)

            metrics = correction_trainer.evaluate(fine_test)
            val_loss = metrics['loss']
            correction_scheduler.step(val_loss)

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
        metrics = correction_trainer.evaluate(fine_test)

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