import torch
import copy
from torch.utils.data import random_split
from aerognn.data.dataset import BuildingDataset
from aerognn.models.gcn_surrogate import BuildingGCN
from torch_geometric.loader import DataLoader
from aerognn.training.trainer import train_epoch, evaluate

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

if __name__ == "__main__":
    train_final_model()

        
