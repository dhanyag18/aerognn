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
    
    cs = generate_cross_section(n_1 = n, n_2 = n, n_3 = n, m=m, num_points = 36)
    cs = apply_aspect_ratio(cs, ar)
    cs = normalize_area(cs)
    verts, faces = extrude_building(
        20, taper=taper, bulge=bulge,
        helical_twist=twist, num_setbacks=setbacks,
        setback_reduction=setback_ratio,
        chamfer_distance=chamfer, cross_section=cs
    )
    
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
def optimize_gnn(num_candidates, top_k):
    import torch
    from aerognn.models.gcn_surrogate import BuildingGCN
    from aerognn.optimization.search import get_diverse_gnn_recommendations
    
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
@click.option('--output-dir', '-o', type=str, default='./results')
def analyze(n, m, ar, twist, bulge, taper, setbacks,
            setback_ratio, chamfer, output_dir):
    """Generate building, predict flow field, and visualize."""
    import os
    import json
    import torch
    import numpy as np
    import pyvista as pv
    from aerognn.data.params_to_graph import params_to_graph
    from aerognn.models.pinn_graphnet import PINNGraphNet
    from aerognn.visualization.pressure_map import plot_surface_pressure
    from aerognn.visualization.streamlines import plot_streamlines
    from aerognn.geometry.superformula import generate_cross_section, apply_aspect_ratio, normalize_area
    from aerognn.geometry.extrusion import extrude_building
    from torch_geometric.loader import DataLoader

    os.makedirs(output_dir, exist_ok=True)

    graph = params_to_graph(n, m, ar, twist, bulge, taper, setbacks, setback_ratio, chamfer)
    loader = DataLoader([graph], batch_size=1, shuffle=False)

    model = PINNGraphNet()
    model.load_state_dict(torch.load("final_pinn_model.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        for batch in loader:
            output = model(batch)

    pressure_vals = output['pressure'].squeeze().numpy()
    velocity_vals = output['velocity'].numpy().astype(np.float32)
    scores = output['scores'][0].tolist()
    cd, cl, cl_std = scores
    composite = 0.2 * cd + 0.2 * abs(cl) + 0.6 * cl_std

    wall_mask = (batch.node_types == 1).numpy()
    wall_pos = batch.pos[wall_mask].numpy()
    wall_pressure = pressure_vals[wall_mask]

    cs = generate_cross_section(n_1=n, n_2=n, n_3=n, m=m, num_points=100)
    cs = apply_aspect_ratio(cs, ar)
    cs = normalize_area(cs)
    verts, faces = extrude_building(
        cross_section=cs, num_layers=50,
        taper=taper, bulge=bulge, helical_twist=twist,
        num_setbacks=setbacks, setback_reduction=setback_ratio,
        chamfer_distance=chamfer
    )

    pressure_path = os.path.join(output_dir, 'pressure_map.png')
    plot_surface_pressure(wall_pos, wall_pressure, verts, faces,
                          title=f'Pressure Distribution (n={n}, m={m})',
                          output_path=pressure_path)

    all_pos = batch.pos.numpy().astype(np.float32)
    pv_faces = np.hstack([np.full((len(faces), 1), 3), faces])
    building_mesh = pv.PolyData(verts.astype(np.float32), pv_faces)
    streamlines_path = os.path.join(output_dir, 'streamlines.png')
    plot_streamlines(all_pos, velocity_vals, building_mesh, output_path=streamlines_path)

    report = {
        'parameters': {
            'n': n, 'm': m, 'ar': ar, 'twist': twist,
            'bulge': bulge, 'taper': taper, 'setbacks': setbacks,
            'setback_ratio': setback_ratio, 'chamfer': chamfer
        },
        'predictions': {
            'cd_mean': cd,
            'cl_mean': cl,
            'cl_std': cl_std,
            'composite_score': composite
        }
    }
    report_path = os.path.join(output_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    click.echo(f'Results saved to {output_dir}/')
    click.echo(f'  pressure_map.png')
    click.echo(f'  streamlines.png')
    click.echo(f'  report.json')
    click.echo(f'Predicted score: {composite:.4f} | Cd: {cd:.4f}, Cl: {cl:.4f}, Cl_std: {cl_std:.4f}')

if __name__ == '__main__':
    cli()