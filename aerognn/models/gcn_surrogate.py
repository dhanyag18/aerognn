import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class BuildingGCN(torch.nn.Module):
    def __init__(self, in_channels=8, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden * 2)
        self.conv3 = GCNConv(hidden * 2, hidden)
        self.lin1 = torch.nn.Linear(hidden, hidden // 2)
        self.lin2 = torch.nn.Linear(hidden // 2, 1)
        self.dropout = torch.nn.Dropout(0.2)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        x = global_mean_pool(x, batch)
        
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x.squeeze(-1)
