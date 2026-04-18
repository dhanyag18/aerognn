import torch
from torch_geometric.utils import scatter

class PhysicsLoss(torch.nn.Module):
    
    def __init__(self, nu=1.5e-5, rho=1.225):
       
        super().__init__()
        self.nu = nu
        self.rho = rho
        
    def compute_gradients(self, f, pos, edge_index):
        
        src, dst = edge_index
        
        delta_pos = pos[dst] - pos[src]  
        dist = torch.norm(delta_pos, dim=1, keepdim=True) + 1e-8
    
        delta_f = f[dst] - f[src]  
        
        weights = 1.0 / dist  
        
        w_dp = weights * delta_pos  
        cov = torch.zeros(pos.shape[0], 3, 3, device=pos.device)
        for d1 in range(3):
            for d2 in range(3):
                cov_contrib = w_dp[:, d1] * delta_pos[:, d2]
                cov[:, d1, d2] = scatter(cov_contrib, src,
                    dim=0, reduce='sum', dim_size=pos.shape[0])
        
        rhs = torch.zeros(pos.shape[0], 3, f.shape[1], device=pos.device)
        for d in range(3):
            val = w_dp[:, d].unsqueeze(1) * delta_f  
            rhs[:, d, :] = scatter(val, src,
                dim=0, reduce='sum', dim_size=pos.shape[0])
        
        cov += 1e-6 * torch.eye(3, device=pos.device).unsqueeze(0)
        grad = torch.linalg.solve(cov, rhs) 
        
        return grad.permute(0, 2, 1)
    
    def continuity_residual(self, velocity, pos, edge_index, node_mask):
        
        grad_vel = self.compute_gradients(velocity, pos, edge_index)
        div = (grad_vel[:, 0, 0]  
             + grad_vel[:, 1, 1]    
             + grad_vel[:, 2, 2])   
        
        return (div[node_mask] ** 2).mean()
    
    def momentum_residual(self, velocity, pressure, pos,
                          edge_index, node_mask):
        
        u, v, w = velocity[:, 0:1], velocity[:, 1:2], velocity[:, 2:3]
        
        grad_vel = self.compute_gradients(velocity, pos, edge_index)
        
        grad_p = self.compute_gradients(pressure, pos, edge_index)
        
        
        conv_x = (u * grad_vel[:, 0:1, 0]
                + v * grad_vel[:, 0:1, 1]
                + w * grad_vel[:, 0:1, 2])
        conv_y = (u * grad_vel[:, 1:2, 0]
                + v * grad_vel[:, 1:2, 1]
                + w * grad_vel[:, 1:2, 2])
        conv_z = (u * grad_vel[:, 2:3, 0]
                + v * grad_vel[:, 2:3, 1]
                + w * grad_vel[:, 2:3, 2])
     
        laplacian_u = self._graph_laplacian(velocity, pos, edge_index)
        
        R_x = conv_x + (1/self.rho)*grad_p[:, :, 0:1] - self.nu*laplacian_u[:, 0:1]
        R_y = conv_y + (1/self.rho)*grad_p[:, :, 1:2] - self.nu*laplacian_u[:, 1:2]
        R_z = conv_z + (1/self.rho)*grad_p[:, :, 2:3] - self.nu*laplacian_u[:, 2:3]
        
        R = torch.cat([R_x, R_y, R_z], dim=1)  
        return (R[node_mask] ** 2).mean()
    
    def boundary_loss(self, velocity, node_types, abl_velocity=None):

        loss = torch.tensor(0.0, device=velocity.device)
        surface_mask = (node_types == 1)
        
        if surface_mask.any():
            loss += (velocity[surface_mask] ** 2).mean()
        
        if abl_velocity is not None:
            inlet_mask = (node_types == 2)
            if inlet_mask.any():
                loss += ((velocity[inlet_mask] - abl_velocity[inlet_mask]) ** 2).mean()
        
        return loss
