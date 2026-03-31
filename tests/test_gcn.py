import torch
import math
import numpy as np
from torch_geometric.loader import DataLoader
from aerognn.models.gcn_surrogate import BuildingGCN
from aerognn.geometry.mesh_to_graph import mesh_to_pyg_graph
from aerognn.training.trainer import cross_validation, train_epoch
from aerognn.data.dataset import BuildingDataset
def test_model():

    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]  
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2, 3], 
        [4, 5, 6, 7], 
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6], 
        [3, 0, 4, 7] 
    ])

    score, cd, cl, cl_std = 0.5, 1.2, 0.1, 0.05

    data = mesh_to_pyg_graph(
        vertices=vertices,
        faces=faces,
        id=1, 
        score=score,
        cd_mean=cd,
        cl_mean=cl,
        cl_std=cl_std

    )

    loader = DataLoader([data], batch_size=32, shuffle=False)

    model = BuildingGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()  

    model.train()
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        assert out.dim() == 1
        loss = criterion(out, data.y[:, 0])
        assert not math.isnan(loss)
        loss.backward()
        optimizer.step()

def test_train_epoch():
    dataset = BuildingDataset()
    loader = DataLoader(dataset[:10], batch_size = 4, shuffle= False)
    model = BuildingGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss() 
    for epoch in range(5):
        loss = train_epoch(model, loader, optimizer, criterion)
        assert not math.isnan(loss)

def test_cross_validate():
    dataset = BuildingDataset()
    mae = cross_validation(dataset, epochs = 1)
    assert not math.isnan(mae)


    

