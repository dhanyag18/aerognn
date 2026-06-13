import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from aerognn.models.uncertainty import mc_dropout_predict
from aerognn.optimization.search import greedy_diverse
from aerognn.training.physics_loss import PhysicsLoss

class AcquisitionFunction:

    def __init__(self, model, physics=None, alpha=0.5, beta=0.1,
                 continuity_threshold=0.1, physics_penalty_weight=1.0):
        self.model = model
        self.physics = physics if physics is not None else PhysicsLoss()
        self.alpha = alpha
        self.beta = beta
        self.continuity_threshold = continuity_threshold
        self.physics_penalty_weight = physics_penalty_weight

    def score_candidate(self, graph):
        result = mc_dropout_predict(self.model, graph, n_samples=20)
        predicted_score = result['score_mean']
        uncertainty = result['score_std']

        loader = DataLoader([graph], batch_size=1)
        with torch.no_grad():
            for batch in loader:
                output = self.model(batch)
                interior_mask = (batch.node_types == 0)
                cont_residual = self.physics.continuity_residual(
                    output['velocity'], batch.pos,
                    batch.edge_index, interior_mask
                )

        acquisition = ((1 - self.alpha - self.beta) * (-predicted_score) + self.alpha * uncertainty + self.beta * cont_residual.item())

        return {
            'acquisition': acquisition,
            'predicted_score': predicted_score,
            'uncertainty': uncertainty,
            'continuity_residual': cont_residual.item(),
        }

    def select_batch(self, candidate_graphs, candidate_params_df, n_recs, features):
        
        all_scores = []
        for graph in candidate_graphs:
            result = self.score_candidate(graph)
            all_scores.append(result['acquisition'])

        candidate_params_df = candidate_params_df.copy()
        candidate_params_df['gnn_preds'] = all_scores
        candidate_params_df['_graph_idx'] = range(len(candidate_graphs))

        scaler = StandardScaler()
        scaler.fit(candidate_params_df[features])

        top_pool = candidate_params_df.nlargest(max(100, len(candidate_params_df) // 10), 'gnn_preds').copy()

        selected_params = greedy_diverse(top_pool, 'gnn_preds', n_recs, scaler, features, ascending=False)

        selected_graphs = [candidate_graphs[i] for i in selected_params['_graph_idx'].tolist()]

        return selected_graphs, selected_params