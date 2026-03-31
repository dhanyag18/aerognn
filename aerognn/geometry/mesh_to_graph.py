import torch
from torch_geometric.data import Data
import numpy as np
import pyvista as pv



def mesh_to_pyg_graph(
    vertices: np.ndarray,
    faces: np.ndarray,
    id: int = None, 
    score: float = None, 
    cd_mean: float = None,
    cl_mean: float = None,
    cl_std: float = None,
    height: float = 200.0
):
    edge_index = _faces_to_edge_index(faces)
    x = _compute_node_features(vertices, faces, height=height)
    
    data = Data(x=x, edge_index=edge_index)
    
    if id is not None:
        data.id = id

    if score is not None:
        data.y = torch.tensor([[
            score, 
            cd_mean if cd_mean is not None else 0.0, 
            cl_mean if cl_mean is not None else 0.0, 
            cl_std if cl_std is not None else 0.0
        ]], dtype=torch.float)

    return data

def _faces_to_edge_index(
        faces: np.ndarray 
):
    edges = set()
    for face in faces:
        n = len(face)
        for i in range(n):
            a, b = int(face[i]), int(face[(i + 1) % n])
            edges.add((min(a, b), max(a, b)))  
    edge_list = []
    for a, b in edges:
        edge_list.append([a, b])
        edge_list.append([b, a])
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    
def _compute_node_features(vertices, faces, height=200.0):
    num_nodes = len(vertices)
    
    pos_norm = vertices.copy()
    pos_norm[:, :2] /= np.max(np.abs(vertices[:, :2]))
    pos_norm[:, 2] /= height
    
    h_frac = (vertices[:, 2] / height).reshape(-1, 1)
    
    dist = np.sqrt(vertices[:, 0]**2 + vertices[:, 1]**2)
    dist = (dist / dist.max()).reshape(-1, 1)
    
    face_counts = np.full((faces.shape[0], 1), faces.shape[1])
    pv_faces = np.c_[face_counts, faces].ravel()
    mesh = pv.PolyData(vertices, pv_faces)
    mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    normals = mesh["Normals"]   
    
    features = np.hstack([pos_norm, h_frac, dist, normals])
    return torch.tensor(features, dtype=torch.float)


    




    
 