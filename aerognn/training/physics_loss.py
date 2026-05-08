import torch
from torch_geometric.utils import scatter


class PhysicsLoss(torch.nn.Module):

    def __init__(self, nu: float = 1.5e-5):
        super().__init__()
        self.nu = nu
    
    def _reliable_mask(self, node_mask, src, N, min_neighbors=6):
        count = torch.zeros(N, device=src.device)
        count.scatter_add_(0, src, torch.ones(src.shape[0], device=src.device))
        return node_mask & (count >= min_neighbors)

    def _compute_gradients(self, f, pos, edge_index):
        src, dst = edge_index
        N = pos.shape[0]

        delta_pos = pos[dst] - pos[src]
        dist = torch.norm(delta_pos, dim=1, keepdim=True).clamp(min=1e-8)
        weights = 1.0 / dist
        delta_f = f[dst] - f[src]
        w_dp = weights * delta_pos

        cov = torch.zeros(N, 3, 3, device=pos.device)
        for d1 in range(3):
            for d2 in range(3):
                contrib = w_dp[:, d1] * delta_pos[:, d2]
                cov[:, d1, d2] = scatter(contrib, src, dim=0, reduce='sum', dim_size=N)
        cov += 1e-6 * torch.eye(3, device=pos.device).unsqueeze(0)

        rhs = torch.zeros(N, 3, f.shape[1], device=pos.device)
        for d in range(3):
            val = w_dp[:, d].unsqueeze(1) * delta_f
            rhs[:, d, :] = scatter(val, src, dim=0, reduce='sum', dim_size=N)

        grad = torch.linalg.solve(cov, rhs)
        return grad.permute(0, 2, 1)

    def _graph_laplacian(self, f, pos, edge_index):
        src, dst = edge_index
        N = pos.shape[0]

        delta_pos = pos[dst] - pos[src]
        dist = torch.norm(delta_pos, dim=1, keepdim=True).clamp(min=1e-8)
        weights = 1.0 / dist
        delta_f = f[dst] - f[src]
        weighted_diff = weights * delta_f

        laplacian = torch.zeros_like(f)
        for d in range(f.shape[1]):
            laplacian[:, d] = scatter(weighted_diff[:, d], src, dim=0, reduce='sum', dim_size=N)
        return laplacian



    def continuity_residual(self, velocity, pos, edge_index, node_mask):
        src, _ = edge_index
        mask = self._reliable_mask(node_mask, src, pos.shape[0])
        if not mask.any():
            return torch.tensor(0.0, device=pos.device)

        grad_vel = self._compute_gradients(velocity, pos, edge_index)
        div = grad_vel[:, 0, 0] + grad_vel[:, 1, 1] + grad_vel[:, 2, 2]
        return (div[mask] ** 2).mean()

    def momentum_residual(self, velocity, pressure, pos, edge_index, node_mask):
        src, _ = edge_index
        N = pos.shape[0]
        mask = self._reliable_mask(node_mask, src, N)
        if not mask.any():
            return torch.tensor(0.0, device=pos.device)

        pressure = pressure.reshape(-1, 1)
        u, v, w = velocity[:, 0:1], velocity[:, 1:2], velocity[:, 2:3]

        grad_vel = self._compute_gradients(velocity, pos, edge_index)
        grad_p = self._compute_gradients(pressure, pos, edge_index)
        lap = self._graph_laplacian(velocity, pos, edge_index)

        conv_x = u * grad_vel[:, 0:1, 0] + v * grad_vel[:, 0:1, 1] + w * grad_vel[:, 0:1, 2]
        conv_y = u * grad_vel[:, 1:2, 0] + v * grad_vel[:, 1:2, 1] + w * grad_vel[:, 1:2, 2]
        conv_z = u * grad_vel[:, 2:3, 0] + v * grad_vel[:, 2:3, 1] + w * grad_vel[:, 2:3, 2]

        R_x = conv_x + grad_p[:, 0, 0:1] - self.nu * lap[:, 0:1]
        R_y = conv_y + grad_p[:, 0, 1:2] - self.nu * lap[:, 1:2]
        R_z = conv_z + grad_p[:, 0, 2:3] - self.nu * lap[:, 2:3]

        R = torch.cat([R_x, R_y, R_z], dim=1)
        return (R[mask] ** 2).mean()
