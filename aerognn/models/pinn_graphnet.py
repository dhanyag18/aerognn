import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class PINNGraphNet(torch.nn.Module):
    def __init__(self, in_channels=14, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden * 2)
        self.conv3 = SAGEConv(hidden * 2, hidden * 2)
        self.conv4 = SAGEConv(hidden * 2, hidden)
        self.flow_lin1 = torch.nn.Linear(hidden, hidden // 2)
        self.flow_lin2 = torch.nn.Linear(hidden // 2, 4)
        self.score_lin1 = torch.nn.Linear(hidden, hidden // 2)
        self.score_lin2 = torch.nn.Linear(hidden // 2, 3)
        self.dropout = torch.nn.Dropout(0.15)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        surface_mask = (data.node_types == 1)

        xf = F.relu(self.conv1(x, edge_index))
        xf = self.dropout(xf)
        xf = F.relu(self.conv2(xf, edge_index))
        xf = self.dropout(xf)
        xf = F.relu(self.conv3(xf, edge_index))
        xf = self.dropout(xf)
        xf = F.relu(self.conv4(xf, edge_index))
        xf = self.dropout(xf)

        flow = F.relu(self.flow_lin1(xf))
        flow = self.flow_lin2(flow)
        velocity = flow[:, :3]
        pressure = flow[:, 3:]

        wall_embeddings = xf[surface_mask]
        wall_batch = batch[surface_mask]
        x_pool = global_mean_pool(wall_embeddings, wall_batch)
        scores = F.relu(self.score_lin1(x_pool))
        scores = self.score_lin2(scores)

        return {
            'velocity': velocity,
            'pressure': pressure,
            'scores': scores,
            'node_embeddings': xf,
        }