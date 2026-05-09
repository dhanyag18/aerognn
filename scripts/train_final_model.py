import torch
import numpy as np
import copy
from torch.utils.data import random_split
from aerognn.data.dataset import BuildingDataset
from aerognn.models.gcn_surrogate import BuildingGCN
from aerognn.models.pinn_graphnet import PINNGraphNet
from aerognn.training.pinn_trainer import PINNTrainer, PhysicsLoss
from torch_geometric.loader import DataLoader
from aerognn.training.trainer import train_epoch, evaluate

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_final_model():
    
    dataset = BuildingDataset()
    length = len(dataset)
    final_model = BuildingGCN()
    
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_x, test_x = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_x, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_x, batch_size=32, shuffle=False)
    
    epochs = 300
    best_val_loss = float('inf')
    best_model_state = None
    patience = 30
    no_improve_count = 0
    
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss() 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    ) 
    
    for i in range(epochs):
        
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
            no_improve_count+=1
        if (no_improve_count >= patience):
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
    
    epochs = 400
    best_val_loss = float('inf')
    best_model_state = None
    patience = 50
    no_improve_count = 0
    curriculum_transitions = {99, 199}
 
    for epoch in range(epochs):
        for batch in train_loader:
            trainer.train_step(batch, epoch)
        
        metrics = trainer.evaluate(test_loader)
        val_loss = metrics['loss']
        scheduler.step(val_loss)
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {val_loss:.4f}")
            
        else:
            metrics = trainer.evaluate(test_loader)
            val_loss = metrics['loss']
            scheduler.step(val_loss)
 
        if epoch in curriculum_transitions:
            
            best_val_loss = float('inf')
            no_improve_count = 0
            optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=20, factor=0.5, min_lr=1e-5)
            trainer.optimizer = optimizer
            print(f"Epoch {epoch+1}: New physics loss introduced, resetting early stopping", flush=True)
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

    
if __name__ == "__main__":
    train_final_pinn_model()

        
