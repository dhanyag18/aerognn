import torch
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader
from sklearn.model_selection import GroupKFold
from aerognn.models.gcn_surrogate import BuildingGCN
from aerognn.data.groups import get_groups
import copy

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    preds, actuals = [], []
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y[:, 0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        preds.extend(pred.tolist())
        actuals.extend(batch.y[:, 0].tolist())
    mae = sum(abs(p-a) for p,a in zip(preds, actuals)) / len(preds)
    return total_loss / len(loader.dataset), mae

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

    BATCH_GROUPS = get_groups()
   
    groups_labels = [BATCH_GROUPS[d.id] for d in dataset]
    cv_strategy = GroupKFold(n_splits = 10)
    fold_mae = []
    fold_r2 = []
    for (train_idx, test_idx) in cv_strategy.split(range(len(dataset)), groups=groups_labels):
        
        train_x = DataLoader([dataset[i] for i in train_idx], batch_size=32, shuffle=False)
        test_x = DataLoader([dataset[i] for i in test_idx], batch_size=32, shuffle=False)
        
        model = BuildingGCN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()  
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = 30
        no_improve_count = 0

        for epoch in range (epochs):
            train_loss, train_mae = train_epoch(model, train_x, optimizer, criterion)
            val_loss, mae, r_squared = evaluate(model, test_x, criterion)
            scheduler.step(val_loss)
            if (epoch + 1) % 50 == 0:
                print(f"Epoch: {epoch + 1}, Training Loss: {train_loss}, Validation loss: {val_loss}, MAE: {mae}, R^2: {r_squared}")
                
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
        val_loss, mae, r2 = evaluate(model, test_x, criterion)
        
        fold_mae.append(mae)
        fold_r2.append(r2)
    
    avg_mae = sum(fold_mae)/len(fold_mae)
    avg_r2 = sum(fold_r2)/len(fold_r2)
    return avg_mae, avg_r2