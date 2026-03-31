import torch
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from sklearn.model_selection import GroupKFold
from aerognn.models.gcn_surrogate import BuildingGCN

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y[:, 0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds, actuals = [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch)
            total_loss += criterion(pred, batch.y[:, 0]).item() * batch.num_graphs
            preds.extend(pred.tolist())
            actuals.extend(batch.y[:, 0].tolist())
    mae = sum(abs(p-a) for p,a in zip(preds, actuals)) / len(preds)
    r_squared = r2_score(actuals, preds)
    return total_loss / len(loader.dataset), mae, r_squared

def cross_validation(dataset, epochs):

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
   
    groups = [BATCH_GROUPS[d.id.item()] for d in dataset]
    cv_strategy = GroupKFold(n_splits = 10)
    fold_mae = []
    fold_r2 = []
    for (train_idx, test_idx) in cv_strategy.split(range(len(dataset)), groups=groups):
        
        train_x = DataLoader([dataset[i] for i in train_idx], batch_size=32, shuffle=False)
        test_x = DataLoader([dataset[i] for i in test_idx], batch_size=32, shuffle=False)
        
        model = BuildingGCN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()  
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5
        )

        for epoch in range (epochs):
            train_loss = train_epoch(model, train_x, optimizer, criterion)
            val_loss, mae, r_squared = evaluate(model, test_x, criterion)
            scheduler.step(val_loss)
            if (epoch + 1) % 50 == 0:
                print(f"Epoch: {epoch + 1}, Training Loss: {train_loss}, Validation loss: {val_loss}, MAE: {mae}, R^2: {r_squared}")
        
        fold_mae.append(mae)
        fold_r2.append(r_squared)
    
    avg_mae = sum(fold_mae)/len(fold_mae)
    avg_r2 = sum(fold_r2)/len(fold_r2)
    return avg_mae, avg_r2