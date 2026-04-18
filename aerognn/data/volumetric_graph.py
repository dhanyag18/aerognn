import torch
import numpy as np
from torch_geometric.data import Data

def build_volumetric_graph(
    cell_centers: np.ndarray,
    cell_neighbors: list,
    boundary_info: dict,
    velocity: np.ndarray,
    pressure: np.ndarray,
    id: int,
    cd_mean: float,
    cl_mean: float,
    cl_std: float,
    height: float=200.0
):
    pos = torch.tensor(cell_centers, dtype=torch.float)
    edge_index = _build_edge_index(cell_neighbors)
    node_types = classify_nodes(cell_centers, boundary_info)
    x = _compute_volumetric_node_features(
        cell_centers=cell_centers,
        node_types=node_types,
        velocity=velocity,
        pressure=pressure,
        height=height
    )
    
    U_ref = 6
    z_ref = 10
    z0 = 0.1
    abl = np.zeros_like(velocity)
    abl[:, 0] = (U_ref / np.log((z_ref + z0) / z0)) * np.log((cell_centers[:, 2] + z0) / z0)
    
    data = Data(x=x, edge_index=edge_index, pos=pos)
    data.node_types = node_types
    data.id = id
    data.y = torch.tensor([[cl_std, cl_mean, cd_mean]], dtype=torch.float)
    data.y_velocity = torch.tensor(velocity, dtype=torch.float)
    data.y_pressure = torch.tensor(pressure, dtype=torch.float)
    data.abl_velocity = torch.tensor(abl, dtype=torch.float)
    
    return data

def _build_edge_index(cell_neighbors: list):
    sources = []
    targets = []
    for cell_id, neighbors in enumerate(cell_neighbors):
        for neighbor_id in neighbors:
            sources.append(cell_id)
            targets.append(neighbor_id)
    return torch.tensor([sources, targets], dtype=torch.long)

def classify_nodes(cell_centers: np.ndarray, boundary_info: dict):
    node_types = torch.zeros(len(cell_centers), dtype=torch.long)
    for idx, btype in boundary_info.items():
        if btype == 'wall':
            node_types[idx] = 1
        elif btype == 'inlet':
            node_types[idx] = 2
        elif btype == 'outlet':
            node_types[idx] = 3
    return node_types

def _compute_volumetric_node_features(
    cell_centers: np.ndarray,
    node_types: torch.Tensor,
    velocity: np.ndarray,
    pressure: np.ndarray,
    height: float
):
    n_cells = len(cell_centers)
    pos_norm = cell_centers.copy()
    pos_norm[:, :2] /= np.abs(cell_centers[:, :2]).max()
    pos_norm[:, 2] /= height
    pos_norm = torch.tensor(pos_norm, dtype=torch.float)
    h_frac = torch.tensor(
        cell_centers[:, 2] / height, dtype=torch.float
    ).unsqueeze(1)
    dist = np.sqrt(cell_centers[:, 0] ** 2 + cell_centers[:, 1] ** 2)
    dist = torch.tensor(dist / dist.max(), dtype=torch.float).unsqueeze(1)
    vel = torch.tensor(velocity, dtype=torch.float)
    pres = torch.tensor(pressure, dtype=torch.float).unsqueeze(1)
    node_type_final = torch.zeros(n_cells, 3)
    node_type_final.scatter_(1, node_types.unsqueeze(1).clamp(max=2), 1)
    return torch.cat([pos_norm, h_frac, dist, vel, pres, node_type_final], dim=1)