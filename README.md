# AeroGNN

AeroGNN is a physics-informed graph neural network system for aerodynamic prediction and design optimization of parametric tall buildings. It replaces hours of CFD simulation with millisecond predictions while enforcing physics constraints.

## Overview

AeroGNN takes 9 building design parameters as input and predicts:

- Mean drag coefficient (Cd)
- Mean lift coefficient (Cl)
- Lift coefficient standard deviation (Clstd)
- Full 3D velocity field (u, v, w)
- Full pressure field at every mesh node
- A composite aerodynamic score: 0.2Cd + 0.2|Cl| + 0.6Clstd

Building geometry is generated using the Gielis superformula, parameterizing cross-sectional shape, aspect ratio, helical twist, bulge, taper, setbacks, and chamfer. All 498 training simulations were run using pimpleFoam (URANS) in OpenFOAM.

## Phases

Phase 1 - Baseline GNN Surrogate
A GNN was trained on 498 OpenFOAM simulations to predict aerodynamic force coefficients from building surface mesh graphs. Compared against XGBoost and Gaussian Process baselines.

Phase 2 - Physics-Informed GNN (PINNGraphNet)
A volumetric graph neural network with dual prediction heads (one for per-node flow field prediction (velocity and pressure) and one for graph-level force coefficient prediction). 

Physics constraints are added directly to the training loss:
    - Continuity residual — enforces mass conservation
    - Momentum residual - enforces steady Navier-Stokes equations

Physics losses are introduced progressively after the model learns from the data.

## Installation

```bash
conda create -n aerognn python=3.11
conda activate aerognn
pip install torch torch-geometric pyvista scikit-learn xgboost trimesh shapely
```
Requires Docker with OpenFOAM container.

## Usage 

Generate building

```bash
python -m aerognn.cli analyze \
  --n 7.7 --m 4 --ar 2.0 --twist 5.0 \
  --bulge 1.0 --taper 0.9 --setbacks 2 \
  --setback-ratio 0.3 --chamfer 3.0 \
  --output-dir ./results
```

Outputs saved to `./results/`:
- `pressure_map.png` - saves surface pressure distribution
- `streamlines.png` — saves velocity streamlines visualization around building
- `report.json` - saves predicted Cd, Cl, Clstd, composite score
