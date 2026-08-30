# AeroGNN

AeroGNN is a multi-fidelity physics-informed graph neural network system for aerodynamic prediction and design optimization of parametric tall buildings. It replaces hours of CFD simulation with millisecond predictions while enforcing physics constraints.

## Overview

AeroGNN takes 9 building design parameters as input and predicts:

- Mean drag coefficient (Cd)
- Mean lift coefficient (Cl)
- Lift coefficient standard deviation (Clstd)
- Full 3D velocity field (u, v, w)
- Full pressure field at every mesh node
- A composite aerodynamic score: 0.2Cd + 0.2|Cl| + 0.6Clstd

Building geometry is generated using the Gielis superformula, parameterizing cross-sectional shape, aspect ratio, helical twist, bulge, taper, setbacks, and chamfer. All training simulations were run using pimpleFoam (URANS with k-omega SST) in OpenFOAM.

## Phases

Phase 1 - Baseline GNN Surrogate
A GNN was trained on 498 OpenFOAM simulations to predict aerodynamic force coefficients from building surface mesh graphs. Compared against XGBoost and Gaussian Process baselines.

Phase 2 - Physics-Informed GNN (PINNGraphNet)
A volumetric graph neural network with dual prediction heads (one for per-node flow field prediction (velocity and pressure) and one for graph-level force coefficient prediction). 

Physics constraints are added directly to the training loss:
    - Continuity residual and momentum residual derived from incompressible Navier-Stokes equations

Physics losses are introduced progressively after the model learns from the data.

Phase 3 - Multi-Fidelity Physics-Informed GNN (AeroGNN)

A volumetric graph neural network combining a coarse-resolution surrogate (trained on inexpensive low-fidelity simulations) with a correction model that learns the discrepancy between coarse and fine outputs on a small paired dataset. An active learning loop expands the paired dataset by targeting designs that are both low-scoring and uncertain, and a differential evolution optimizer uses the final model to identify VIV-minimizing geometries. Uncertainty is quantified via MC Dropout and a physics-based continuity residual.

**The final AeroGNN model is included in this repo (`final_mf_model.pt`)**

**Result:** 364× faster than fine-resolution CFD (11.86s vs. 4320.48s per design), with 
cross-validated R² = 0.849 for the aerodynamic score. 

## Installation (Surface GNN)

```bash
conda create -n aerognn python=3.11
conda activate aerognn
pip install torch torch-geometric pyvista scikit-learn xgboost trimesh shapely
```
Requires Docker with OpenFOAM container.

## Usage 

Generate building (Surface GNN)

```bash
python -m aerognn.cli analyze \
  --n 7.7 --m 4 --ar 2.0 --twist 5.0 \
  --bulge 1.0 --taper 0.9 --setbacks 2 \
  --setback-ratio 0.3 --chamfer 3.0 \
  --output-dir ./results
```

## Results

Cross-validated R-squared across model configurations (10-fold GroupKFold). 

| Model | Score | Cd | Cl | Clstd | u | v | w | Pressure |
|---|---|---|---|---|---|---|---|---|
| GP | 0.253 | – | – | – | – | – | – | – |
| XGBoost | 0.516 | – | – | – | – | – | – | – |
| Surface GNN | 0.785 | – | – | – | – | – | – | – |
| PINNGraphNet | 0.763 | 0.769 | 0.702 | 0.208 | 0.349 | 0.418 | 0.020 | 0.561 |
| Coarse only | 0.744 | 0.740 | 0.699 | 0.389 | 0.350 | 0.497 | 0.128 | 0.564 |
| AeroGNN (final) | 0.849 | 0.813 | 0.792 | 0.579 | 0.168 | 0.265 | 0.069 | 0.262 |


