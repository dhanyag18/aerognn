import torch
from aerognn.models.gcn_surrogate import BuildingGCN
from aerognn.data.dataset import BuildingDataset
from aerognn.training.trainer import train_epoch
import numpy as np
from torch_geometric.loader import DataLoader

def train_and_evaluate(dataset, epochs):
    model = BuildingGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    data = DataLoader(dataset, batch_size=32)
    for epoch in range(epochs):
            loss, mae = train_epoch(model, data, optimizer, criterion)
            
    return mae

    
def permutation_test():
    dataset = BuildingDataset()
    
    y = np.array([data.y[0, 0].item() for data in dataset])
    n_permutations = 100
    original_labels = [d.y[0, 0].item() for d in dataset]
    epochs = 20
    
    real_mae = train_and_evaluate(dataset, epochs)
    shuffled_maes = []
    for i in range(n_permutations):
        y_shuffled = np.random.permutation([d.y[0, 0].item() for d in dataset])
        for j, data in enumerate(dataset):
            data.y[0, 0] = y_shuffled[j]
        mae = train_and_evaluate(dataset, epochs)
        shuffled_maes.append(mae)
        print(f"Permutation {i+1}/{n_permutations}: MAE={mae:.4f}")
    
    for j, data in enumerate(dataset):
        data.y[0, 0] = original_labels[j]

    p_value = (sum(1 for m in shuffled_maes if m <= real_mae) + 1) / (n_permutations + 1)
    return real_mae, shuffled_maes, p_value


if __name__ == "__main__":
    real_mae, shuffled_maes, p_value = permutation_test()
    print(f'Real MAE:{real_mae}, Shuffled MAE: {np.mean(shuffled_maes)}, P-value: {p_value}')





