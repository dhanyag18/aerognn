import torch
from aerognn.data.dataset import BuildingDataset
from aerognn.models.gcn_surrogate import BuildingGCN
from torch_geometric.loader import DataLoader
from aerognn.training.trainer import train_epoch

def train_final_model():
    
    dataset = BuildingDataset()
    final_model = BuildingGCN()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    epochs = 200
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss() 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    ) 
    
    for i in range(epochs):
        loss = train_epoch(final_model, loader, optimizer, criterion)
        scheduler.step(loss)
        if (i + 1) % 20 == 0:
            print(f"Epoch {i+1}, Loss: {loss}")

    
    torch.save(final_model.state_dict(), "final_model.pt")
    print("Model saved to final_model.pt")

if __name__ == "__main__":
    train_final_model()

        
