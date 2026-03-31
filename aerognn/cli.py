import click
import torch



@click.group()
def cli():
    """AeroGNN: ML-powered aerodynamic building design."""
    pass


@cli.command()
@click.option('--n', type=float, required=True, help='Shape exponent')
@click.option('--m', type=int, required=True, help='Symmetry parameter')
@click.option('--ar', type=float, default=1.0, help='Aspect ratio')
@click.option('--twist', type=float, default=0.0, help='Helical twist deg')
@click.option('--bulge', type=float, default=1.0, help='Bulge factor')
@click.option('--taper', type=float, default=1.0, help='Taper factor')
@click.option('--setbacks', type=int, default=0, help='Number of setbacks')
@click.option('--setback-ratio', type=float, default=0.2)
@click.option('--chamfer', type=float, default=0.0, help='Chamfer distance')

def generate(n, m, ar, twist, bulge, taper, setbacks, setback_ratio,
             chamfer):
    """Generate a building geometry and predict its score."""
    from torch_geometric.loader import DataLoader
    from aerognn.geometry.superformula import (
        generate_cross_section, apply_aspect_ratio, normalize_area
    )
    from aerognn.geometry.extrusion import extrude_building
    from aerognn.geometry.mesh_to_graph import mesh_to_pyg_graph
    from aerognn.models.gcn_surrogate import BuildingGCN
    
    # Generate geometry
    cs = generate_cross_section(n_1 = n, n_2 = n, n_3 = n, m=m, num_points = 36)
    cs = apply_aspect_ratio(cs, ar)
    cs = normalize_area(cs)
    verts, faces = extrude_building(
        20, taper=taper, bulge=bulge,
        helical_twist=twist, num_setbacks=setbacks,
        setback_reduction=setback_ratio,
        chamfer_distance=chamfer, cross_section=cs
    )
    
    # Convert to graph and predict
    graph = mesh_to_pyg_graph(verts, faces)
    loader = DataLoader([graph], batch_size=1, shuffle=False)
    
    model = BuildingGCN()
    model.load_state_dict(torch.load("final_model.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        for batch in loader:
            score = model(batch)
    
    click.echo(f'Predicted score: {score.item():.4f}')


@cli.command()
@click.option('--num-candidates', type=int, default=1000)
@click.option('--top-k', type=int, default=5)
def optimize(num_candidates, top_k):
    """Search for optimal building designs using the GNN."""

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import pdist
    from aerognn.geometry.superformula import (
        generate_cross_section, apply_aspect_ratio, normalize_area
    )
    from aerognn.geometry.extrusion import extrude_building
    from aerognn.geometry.mesh_to_graph import mesh_to_pyg_graph
    from aerognn.models.gcn_surrogate import BuildingGCN
    from torch_geometric.loader import DataLoader

    
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
            cs = generate_cross_section(n_1=row['n'], n_2= row['n'], n_3 = row['n'], m=int(row['m']), num_points = 36)
            cs = apply_aspect_ratio(cs, row['AR'])
            cs = normalize_area(cs)
            verts, faces = extrude_building(
                num_layers = 20,
                taper=row['taper'],
                bulge=row['bulge'],
                helical_twist=row['helical_twist'],
                num_setbacks=int(row['num_setbacks']),
                setback_reduction=row['setback_reduction'],
                chamfer_distance=row['chamfer_dist'],
                cross_section = cs
            )
            graph = mesh_to_pyg_graph(verts, faces, id = idx)
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
    
    
    click.echo(f'Top {top_k} designs found:')

    safe_ranges = {
        'n': (0.5, 12),
        'm': (1, 12),
        'AR': (0.5, 8),
        'helical_twist': (-360, 360),
        'bulge': (0.5, 2),
        'taper': (0.5, 2),
        'num_setbacks': (0, 3),
        'setback_reduction': (0.05, 0.65),
        'chamfer_dist': (0, 15)
    }
    features = ['n', 'm', 'AR', 'helical_twist', 'bulge', 'taper', 'num_setbacks', 'setback_reduction', 'chamfer_dist']
    int_features_set = {"m", "num_setbacks"}
    
    model = BuildingGCN()
    model.load_state_dict(torch.load("final_model.pt", weights_only=True))
    model.eval()
    top_designs = get_diverse_gnn_recommendations(model, features, int_features_set, safe_ranges, top_k, num_candidates)
    print(top_designs)

if __name__ == '__main__':
    cli()
