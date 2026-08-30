import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class CoarseModel(torch.nn.Module):

    def __init__(self, in_channels=14, hidden=48):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden * 2)
        self.conv3 = SAGEConv(hidden * 2, hidden)
        self.flow_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 4)  
        )
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 3) 
        )
        self.dropout = torch.nn.Dropout(0.15)
    
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = self.dropout(F.relu(self.conv1(x, ei)))
        x = self.dropout(F.relu(self.conv2(x, ei)))
        x = F.relu(self.conv3(x, ei))
        
        flow = self.flow_head(x)
        wall = (data.node_types == 1)
        pooled = global_mean_pool(x[wall], batch[wall])
        scores = self.score_head(pooled)
        return {'velocity': flow[:,:3], 'pressure': flow[:,3:],
                'scores': scores, 'embeddings': x}

class CorrectionModel(torch.nn.Module):
   
    def __init__(self, in_channels=14, coarse_channels=4, hidden=32):
        super().__init__()
        combined = in_channels + coarse_channels
        self.conv1 = SAGEConv(combined, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        
        self.flow_correction = torch.nn.Sequential(
            torch.nn.Linear(hidden, 4)  
        )
        self.score_correction = torch.nn.Sequential(
            torch.nn.Linear(hidden, 3)  
        )
        self.dropout = torch.nn.Dropout(0.15)
    
    def forward(self, data, coarse_output):
        coarse_flow = torch.cat([
            coarse_output['velocity'],
            coarse_output['pressure']
        ], dim=1).detach()  
        
        x = torch.cat([data.x, coarse_flow], dim=1)
        ei, batch = data.edge_index, data.batch
        
        x = self.dropout(F.relu(self.conv1(x, ei)))
        x = F.relu(self.conv2(x, ei))
        
        flow_delta = self.flow_correction(x)
        wall = (data.node_types == 1)
        pooled = global_mean_pool(x[wall], batch[wall])
        score_delta = self.score_correction(pooled)
        
        return {
            'velocity': coarse_output['velocity'] + flow_delta[:,:3],
            'pressure': coarse_output['pressure'] + flow_delta[:,3:],
            'scores': coarse_output['scores'] + score_delta,
        }

class MultiFidelityPINNGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse_model = CoarseModel()
        self.correction_model = CorrectionModel()
    
    def forward(self, data, apply_correction=True):
        coarse_out = self.coarse_model(data)
        if apply_correction:
            corrected = self.correction_model(data, coarse_out)
            return corrected
        return coarse_out
