from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import torch
import numpy as np


def interpolate_fine_to_coarse(fine_centers, fine_values, coarse_centers, coarse_values, wall_mask):
   
    result = coarse_values.copy()  
    nn_interp = NearestNDInterpolator(fine_centers, fine_values)
    result[wall_mask] = nn_interp(coarse_centers[wall_mask])
    
    interior_mask = ~wall_mask
    lin_interp = LinearNDInterpolator(fine_centers, fine_values)
    interior_vals = lin_interp(coarse_centers[interior_mask])
    
    valid = ~np.isnan(interior_vals).any(axis=1) if interior_vals.ndim > 1 \
            else ~np.isnan(interior_vals)
    interior_indices = np.where(interior_mask)[0]
    result[interior_indices[valid]] = interior_vals[valid]
    
    n_filled = (~valid).sum()
    
    return result