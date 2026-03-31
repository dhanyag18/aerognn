import argparse
import pandas as pd
import torch
import os
from aerognn.geometry.superformula import generate_cross_section, apply_aspect_ratio, normalize_area
from aerognn.geometry.extrusion import extrude_building
from aerognn.geometry.mesh_to_graph import mesh_to_pyg_graph



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to simulation results CSV')
    parser.add_argument('--output', default='data/processed/', help='Output directory')
    args = parser.parse_args()
 
    df = pd.read_csv(args.csv)
    os.makedirs(args.output, exist_ok=True)
 
    for idx, row in df.iterrows():
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
 
        graph = mesh_to_pyg_graph(verts, faces, score=row['score'], id = row['id'], cd_mean = row['cd_mean'], cl_mean = row['cl_mean'], cl_std = row['cl_std'] )
        torch.save(graph, os.path.join(args.output, f'building_{idx:04d}.pt'))
        print(f'Converted {idx+1}/{len(df)}')

if __name__ == '__main__':
    main()