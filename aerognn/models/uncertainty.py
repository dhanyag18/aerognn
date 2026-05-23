import torch
import numpy as np

def mc_dropout_predict(model, graph, n_samples=30):
    
    model.train()  
    from torch_geometric.loader import DataLoader
    loader = DataLoader([graph], batch_size=1)
    
    score_samples = []
    velocity_samples = []
    pressure_samples = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            for batch in loader:
                output = model(batch)
                scores = output['scores'][0]
                composite = (0.2 * scores[0] + 0.2 * torch.abs(scores[1])
                           + 0.6 * scores[2])
                score_samples.append(composite.item())
                velocity_samples.append(
                    output['velocity'].numpy().copy())
                pressure_samples.append(
                    output['pressure'].numpy().copy())
    
    model.eval()  
    
    return {
        'score_mean': np.mean(score_samples),
        'score_std': np.std(score_samples),
        'velocity_mean': np.mean(velocity_samples, axis=0),
        'velocity_std': np.std(velocity_samples, axis=0),
        'pressure_mean': np.mean(pressure_samples, axis=0),
        'pressure_std': np.std(pressure_samples, axis=0),
    }
