import torch
from sklearn.model_selection import permutation_test_score, GroupKFold
from aerognn.models.gcn_surrogate import BuildingGCN
from aerognn.data.dataset import BuildingDataset
from aerognn.training.trainer import cross_validation, train_epoch, evaluate
import numpy as np
import copy
from torch_geometric.loader import DataLoader

def train_and_evaluate(dataset, epochs):
    model = BuildingGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss() 

    indices = np.random.permutation(len(dataset))
    split = int(0.8 * len(dataset))
    train_idx = indices[:split]
    test_idx = indices[split:]
    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=32)
    test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=32)
    
    for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion)
    
    val_loss, mae, r_squared = evaluate(model, test_loader, criterion)
    return mae

    
def permutation_test():
    dataset = BuildingDataset()
    
    y = np.array([data.y[0, 0].item() for data in dataset])
    n_permutations = 5
    dataset_shuffled = copy.deepcopy(dataset)
    total_mae = []
    epochs = 150
    
    real_mae = train_and_evaluate(dataset, epochs)
    
    for i in range(n_permutations):
        print(f"Permutation {i+1}/{n_permutations}")
        y_shuffled = np.random.permutation(y)
        for j, data in enumerate(dataset_shuffled):
            data.y[0, 0] = y_shuffled[j]
        
        mae = train_and_evaluate(dataset_shuffled, epochs)
        total_mae.append(mae)

    C = 0
    for i in range(len(total_mae)):
        if total_mae[i] <= real_mae:
            C+=1
    
    pvalue = (C+1)/(n_permutations+1)

    print(f"P-value: {pvalue}")

if __name__ == "__main__":
     permutation_test()






