import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

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
    
    U_ref = 6
    z_ref = 10
    z0 = 0.1
    abl = np.zeros_like(velocity)
    abl[:, 0] = (U_ref/np.log((z_ref + z0)/z0)) * np.log((cell_centers[:, 2] + z0)/z0)
    
    x = _compute_volumetric_node_features(
        cell_centers=cell_centers,
        node_types=node_types,
        abl_velocity=abl,
        height=height
    )
    
    data = Data(x=x, edge_index=edge_index, pos=pos)
    data.node_types = node_types
    data.id = id
    data.y_coeffs = torch.tensor([[cd_mean, cl_mean, cl_std]], dtype=torch.float)
    data.y_velocity = torch.tensor(velocity, dtype=torch.float)
    data.y_pressure = torch.tensor(pressure, dtype=torch.float)
    data.abl_velocity = torch.tensor(abl, dtype=torch.float)
    
    return data

def _compute_point_normals(points: np.ndarray, k: int = 10):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    _, indices = nbrs.kneighbors(points)
    normals = np.zeros_like(points)
    for i, neighbors in enumerate(indices):
        neighborhood = points[neighbors[1:]]
        centered = neighborhood - neighborhood.mean(axis=0)
        cov = centered.T @ centered
        _, _, vh = np.linalg.svd(cov)
        normals[i] = vh[-1]
    return normals

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
        elif btype in ('inlet', 'outlet', 'symmetry'):
            node_types[idx] = 2
    return node_types

def _compute_volumetric_node_features(
    cell_centers: np.ndarray,
    node_types: torch.Tensor,
    abl_velocity: np.ndarray,
    height: float
):
    n_cells = len(cell_centers)
    
    wall_mask = (node_types == 1).numpy()
    if wall_mask.sum() > 0:
        wall_centers = cell_centers[wall_mask]
        xy_scale = np.abs(wall_centers[:, :2]).max()
        building_center_xy = wall_centers[:, :2].mean(axis=0)
    else:
        xy_scale = np.abs(cell_centers[:, :2]).max()
        building_center_xy = np.array([0.0, 0.0])
    
    pos_norm = cell_centers.copy()
    pos_norm[:, :2] /= xy_scale
    pos_norm[:, 2] /= height
    pos_norm = torch.tensor(pos_norm, dtype=torch.float)
    
    h_frac = torch.tensor(
        cell_centers[:, 2] / height, dtype=torch.float
    ).unsqueeze(1)
    
    dx = cell_centers[:, 0] - building_center_xy[0]
    dy = cell_centers[:, 1] - building_center_xy[1]
    raw_dist = np.sqrt(dx**2 + dy**2)
    dist = torch.tensor(raw_dist / (xy_scale + 1e-8), dtype=torch.float).unsqueeze(1)

    wall_normals = np.zeros((n_cells, 3))
    if wall_mask.sum() > 0:
        wall_pts = cell_centers[wall_mask]
        wall_normals[wall_mask] = _compute_point_normals(wall_pts, k=10)
    normals = torch.tensor(wall_normals, dtype=torch.float)

    node_type_final = torch.zeros(n_cells, 3)
    node_type_final.scatter_(1, node_types.unsqueeze(1).clamp(max=2), 1)
    
    abl = torch.tensor(abl_velocity, dtype=torch.float)
    
    return torch.cat([pos_norm, h_frac, dist, node_type_final, abl, normals], dim=1)

def subsample_mesh(cell_centers, cell_neighbors, boundary_info,
                   velocity, pressure,
                   max_wall_nodes, max_interior_nodes):

    n_cells = len(cell_centers)
    
    all_building_ids = np.array([idx for idx, btype in boundary_info.items() if btype == 'wall'])

    if len(all_building_ids) > max_wall_nodes:
        building_center = cell_centers[all_building_ids].mean(axis=0)
        wall_dists = np.linalg.norm(cell_centers[all_building_ids] - building_center, axis=1)
        nearest_wall = np.argsort(wall_dists)[:max_wall_nodes]
        building_ids = set(all_building_ids[nearest_wall].tolist())
    else:
        building_ids = set(all_building_ids.tolist())

    all_building_set = set(all_building_ids.tolist())
    interior_ids = np.array([i for i in range(n_cells) if i not in all_building_set])

    building_center = cell_centers[list(building_ids)].mean(axis=0)
    dists = np.linalg.norm(cell_centers[interior_ids] - building_center, axis=1)
    nearest_idx = np.argsort(dists)[:max_interior_nodes]
    interior_keep = interior_ids[nearest_idx].tolist()
    keep = sorted(list(building_ids) + interior_keep)
    old_to_new = np.full(n_cells, -1, dtype=np.int32)
    for new_id, old_id in enumerate(keep):
        old_to_new[old_id] = new_id

    new_centers = cell_centers[keep]
    new_velocity = velocity[keep]
    new_pressure = pressure[keep]

    new_neighbors = [[] for _ in range(len(keep))]
    for new_id, old_id in enumerate(keep):
        neighbors = np.array(cell_neighbors[old_id], dtype=np.int32)
        if len(neighbors) == 0:
            continue
        mapped = old_to_new[neighbors]
        new_neighbors[new_id] = mapped[mapped >= 0].tolist()

    new_boundary = {}
    for old_id, btype in boundary_info.items():
        new_id = old_to_new[old_id]
        if new_id >= 0:
            new_boundary[int(new_id)] = btype

    return new_centers, new_neighbors, new_boundary, new_velocity, new_pressure