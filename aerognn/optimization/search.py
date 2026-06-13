import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from aerognn.geometry.extrusion import extrude_building
from aerognn.data.params_to_graph import params_to_graph
from torch_geometric.loader import DataLoader
from aerognn.geometry.superformula import (
    generate_cross_section, apply_aspect_ratio, normalize_area
)
from aerognn.geometry.mesh_to_graph import mesh_to_pyg_graph
from aerognn.training.physics_loss import PhysicsLoss

def greedy_diverse(pool_df, score_col, n_pick, scaler_top, feats,
                   ascending=True, min_dist_start=1.5):
    sorted_pool = pool_df.sort_values(score_col, ascending=ascending).reset_index(drop=True)
    scaled_top = scaler_top.transform(sorted_pool[feats])
    for min_d in [min_dist_start, 1.0, 0.5, 0.2]:  
        selected_idx = [0]  
        for i in range(1, len(sorted_pool)):
            if len(selected_idx) >= n_pick:
                break
            dists = [np.sqrt(np.sum((scaled_top[i] - scaled_top[j])**2))
                    for j in selected_idx]
            if min(dists) > min_d:
                selected_idx.append(i)
        if len(selected_idx) >= n_pick:
            break
    return sorted_pool.iloc[selected_idx[:n_pick]]


def get_diverse_gnn_recommendations(model, features, int_features_set, safe_ranges, n_recs, n_candidates, seed=42):
    model.eval()
    rng = np.random.RandomState(seed)
    candidates = pd.DataFrame()
    for f in features:
        lo, hi = safe_ranges[f]
        vals = rng.uniform(lo, hi, n_candidates)
        if f in int_features_set:
            vals = np.round(vals).astype(int).clip(int(lo), int(hi))
        candidates[f] = vals
    
    graphs = []
    for idx, row in candidates.iterrows():
        cs = generate_cross_section(n_1=row['n'], n_2=row['n'], n_3=row['n'], m=int(row['m']), num_points=36)
        cs = apply_aspect_ratio(cs, row['AR'])
        cs = normalize_area(cs)
        verts, faces = extrude_building(
            num_layers=20,
            taper=row['taper'],
            bulge=row['bulge'],
            helical_twist=row['helical_twist'],
            num_setbacks=int(row['num_setbacks']),
            setback_reduction=row['setback_reduction'],
            chamfer_distance=row['chamfer_dist'],
            cross_section=cs
        )
        graph = mesh_to_pyg_graph(verts, faces, id=idx)
        graphs.append(graph)
    
    dataset = DataLoader(graphs, batch_size=32, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for batch in dataset:
            pred = model(batch)
            all_preds.extend(pred.tolist())
    
    candidates["gnn_preds"] = all_preds
    
    scaler = StandardScaler()
    scaler.fit(candidates[features])
    top_pool = candidates.nsmallest(max(1000, n_candidates // 100), "gnn_preds").copy()
    selected = greedy_diverse(top_pool, "gnn_preds", n_recs, scaler, features, ascending=True)
    sel_scaled = scaler.transform(selected[features])
    
    if len(selected) > 1:
        print(f"Diversity: min pairwise distance = {np.min(pdist(sel_scaled)):.2f}")
    return selected

def get_diverse_gnn_pinn_recommendations(
    model, features, int_features_set, safe_ranges, 
    n_recs, n_candidates, 
    physics_penalty_weight=1.0,
    continuity_threshold=0.1,
    seed=42
):
    model.eval()
    physics = PhysicsLoss()
    rng = np.random.RandomState(seed)
    
    candidates = pd.DataFrame()
    for f in features:
        lo, hi = safe_ranges[f]
        vals = rng.uniform(lo, hi, n_candidates)
        if f in int_features_set:
            vals = np.round(vals).astype(int).clip(int(lo), int(hi))
        candidates[f] = vals

    graphs = []
    for idx, row in candidates.iterrows():
        graph = params_to_graph(
            row['n'], row['m'], row['AR'], row['taper'], 
            row['helical_twist'], row['bulge'], 
            row['num_setbacks'], row['setback_reduction'], 
            row['chamfer_dist']
        )
        graphs.append(graph)

    dataset = DataLoader(graphs, batch_size=8, shuffle=False)  
    
    all_scores = []
    
    with torch.no_grad():
        for batch in dataset:
            output = model(batch)
        
            scores = output['scores'] 
            composite = 0.2 * scores[:, 0] + 0.2 * torch.abs(scores[:, 1]) + 0.6 * scores[:, 2]
            
            interior_mask = (batch.node_types == 0)
            cont_residual = physics.continuity_residual(
                output['velocity'], 
                batch.pos, 
                batch.edge_index, 
                interior_mask
            )
            
            if cont_residual.item() > continuity_threshold:
                penalty = physics_penalty_weight * cont_residual.item()
            else:
                penalty = 0.0
            
            penalized_scores = composite + penalty
            all_scores.extend(penalized_scores.tolist())
    
    candidates["gnn_preds"] = all_scores

    scaler = StandardScaler()
    scaler.fit(candidates[features])
    top_pool = candidates.nsmallest(max(1000, n_candidates // 100), "gnn_preds").copy()
    selected = greedy_diverse(top_pool, "gnn_preds", n_recs, scaler, features, ascending=True)
    
    sel_scaled = scaler.transform(selected[features])
    if len(selected) > 1:
        print(f"Diversity: min pairwise distance = {np.min(pdist(sel_scaled)):.2f}")
    
    return selected

