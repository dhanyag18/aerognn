import torch
from aerognn.training.physics_loss import PhysicsLoss

def make_grid_graph(n=5):
    coords = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                coords.append([i*0.1, j*0.1, k*0.1])
    pos = torch.tensor(coords, dtype=torch.float)
    sources, targets = [], []
    for idx in range(len(coords)):
        i = idx // (n * n)
        j = (idx % (n * n)) // n
        k = idx % n
        for di, dj, dk in [
            (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)
        ]:
            ni, nj, nk = i+di, j+dj, k+dk
            if 0 <= ni < n and 0 <= nj < n and 0 <= nk < n:
                neighbor_idx = ni*(n*n) + nj*n + nk
                sources.append(idx)
                targets.append(neighbor_idx)
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    return pos, edge_index

def test_uniform_flow_continuity():
    physics = PhysicsLoss()
    pos,edge_index = make_grid_graph()
    n_nodes = pos.shape[0]
    velocity = torch.ones(n_nodes, 3) * 5.0
    node_mask = torch.ones(n_nodes, dtype=torch.bool)
    residual = physics.continuity_residual(velocity, pos, edge_index, node_mask)
    assert residual.item() < 1e-3

def test_known_gradient():
    physics = PhysicsLoss()
    pos, edge_index = make_grid_graph()
    f = 2.0 * pos[:, 0:1] + 3.0 * pos[:, 1:2] - 1.0 * pos[:, 2:3]
    grad = physics._compute_gradients(f, pos, edge_index)
    interior = torch.zeros(pos.shape[0], dtype=torch.bool)
    interior[12:-12] = True
    grad_interior = grad[interior]
    expected = torch.tensor([2.0, 3.0, -1.0])
    mean_grad = grad_interior[:, 0, :].mean(dim=0)
    assert torch.allclose(mean_grad, expected, atol=0.1)

def test_divergence_free_field():
    physics = PhysicsLoss()
    pos, edge_index = make_grid_graph()
    n_nodes = pos.shape[0]
    velocity = torch.zeros(n_nodes, 3)
    velocity[:, 0] = pos[:, 1]
    velocity[:, 1] = -pos[:, 0]
    velocity[:, 2] = 0.0
    node_mask = torch.ones(n_nodes, dtype=torch.bool)
    residual = physics.continuity_residual(velocity, pos, edge_index, node_mask)
    assert residual.item() < 1e-3

def test_uniform_flow_momentum():
    physics = PhysicsLoss()
    pos, edge_index = make_grid_graph()
    n_nodes = pos.shape[0]
    velocity = torch.ones(n_nodes,3) * 5.0
    pressure = torch.ones(n_nodes,1) * 101325.0
    node_mask = torch.ones(n_nodes, dtype=torch.bool)
    residual = physics.momentum_residual(velocity, pressure, pos, edge_index, node_mask)
    assert residual.item() < 1e-2