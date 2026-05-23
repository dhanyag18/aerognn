import os
import torch
import pandas as pd
from aerognn.data.flow_field_loader import load_openfoam_field
from aerognn.data.volumetric_graph import build_volumetric_graph, subsample_mesh

RAW_DIR = "data/coarse_raw"
OUT_DIR = "data/coarse"
SCORES_CSV = "data/coarse_raw/coarse_results.csv"

os.makedirs(OUT_DIR, exist_ok=True)
scores_df = pd.read_csv(SCORES_CSV, index_col='id')

for case_id in sorted(os.listdir(RAW_DIR), key=lambda x: int(x) if x.isdigit() else float('inf')):
    case_dir = os.path.join(RAW_DIR, case_id)
    if not os.path.isdir(case_dir):
            continue
        
    row = scores_df.loc[int(case_id)]
    fields = load_openfoam_field(case_dir)
    
    centers, neighbors, boundary, vel, pres = subsample_mesh(
        fields['cell_centers'], fields['cell_neighbors'],
        fields['boundary_info'], fields['velocity'], fields['pressure'],
        max_wall_nodes=3000,
        max_interior_nodes=2000
    )
    
    data = build_volumetric_graph(
        cell_centers=centers,
        cell_neighbors=neighbors,
        boundary_info=boundary,
        velocity=vel,
        pressure=pres,
        id=int(case_id),
        cd_mean=float(row['cd_mean']),
        cl_mean=float(row['cl_mean']),
        cl_std=float(row['cl_std']),
    )
    
    out_path = os.path.join(OUT_DIR, f"{case_id}.pt")
    torch.save(data, out_path)
    print(f"Saved {out_path}")