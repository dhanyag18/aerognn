import os
import torch
import pandas as pd
from aerognn.data.flow_field_loader import load_openfoam_field
from aerognn.data.volumetric_graph import build_volumetric_graph

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
SCORES_CSV = "data/raw/simulations.csv"

os.makedirs(OUT_DIR, exist_ok=True)
scores_df = pd.read_csv(SCORES_CSV, index_col='id')

for case_id in sorted(os.listdir(RAW_DIR), key=lambda x: int(x) if x.isdigit() else float('inf')):    
    case_dir = os.path.join(RAW_DIR, case_id)
    if not os.path.isdir(case_dir):
        continue

    row = scores_df.loc[int(case_id)]
    fields = load_openfoam_field(case_dir)

    data = build_volumetric_graph(
        cell_centers=fields['cell_centers'],
        cell_neighbors=fields['cell_neighbors'],
        boundary_info=fields['boundary_info'],
        velocity=fields['velocity'],
        pressure=fields['pressure'],
        id=int(case_id),
        cd_mean=float(row['cd_mean']),
        cl_mean=float(row['cl_mean']),
        cl_std=float(row['cl_std']),
    )

    out_path = os.path.join(OUT_DIR, f"{case_id}.pt")
    torch.save(data, out_path)
    print(f"Saved {out_path}")