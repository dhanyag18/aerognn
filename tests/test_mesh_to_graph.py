import numpy as np
import torch 
from aerognn.geometry.mesh_to_graph import mesh_to_pyg_graph
def test_cube():
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

    assert data.num_nodes == 8
    assert data.edge_index.shape[1] == 24
    assert data.x.shape == (8,8)
    assert data.edge_index.min() >= 0
    assert data.edge_index.max() < data.num_nodes
    edges = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    for (i, j) in edges:
        assert (j, i) in edges
    actual = torch.tensor([[0.5, 1.2, 0.1, 0.05]], dtype=torch.float)
    assert data.y.shape == (1, 4)
    assert torch.allclose(data.y, actual)


