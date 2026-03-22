import numpy as np

def extrude_building(
        num_layers,
        taper,
        bulge,
        helical_twist,
        num_setbacks,
        setback_reduction,
        chamfer_distance,
        cross_section: np.ndarray,
        height: float = 200.0   
):
    
    layers = np.linspace(0, height, num_layers)
    full_layers= []
    faces = []
    n_points = 360

    for i in range(num_layers):
        layer_coords=cross_section.copy()
        factor = i/(num_layers-1)
        z=layers[i]

        if num_setbacks > 0:
            tiers = int(factor * (num_setbacks+1))
            setback_scale = (1 - setback_reduction) ** tiers
        else:
            setback_scale = 1

        
        taper_scale = 1 + (taper - 1) * factor
        
        bulge_scale = 1 + (bulge - 1) * np.sin(np.pi * factor)
       
        helical_angle = np.radians(factor * helical_twist)
        helical_scale = np.array([[np.cos(helical_angle), -np.sin(helical_angle)], [np.sin(helical_angle), np.cos(helical_angle)]])

        layer_coords = (layer_coords * taper_scale * bulge_scale) @ (helical_scale) * setback_scale
        if chamfer_distance > 0:
            norm=np.linalg.norm(layer_coords, axis=1)
            max_r=np.max(norm)
            threshold =max_r-chamfer_distance
            scale_factors = np.where(norm>threshold, threshold/norm, 1.0)
            layer_coords=layer_coords*scale_factors[:, np.newaxis]
        
        
        if i > 0:
            curr_start = i * n_points
            prev_start = (i - 1) * n_points
            for j in range(n_points):
                v1 = prev_start + j
                v2 = prev_start + (j + 1) % n_points
                v3 = curr_start + j
                v4 = curr_start + (j + 1) % n_points
                faces.append([v2, v1, v3])
                faces.append([v4, v2, v3])

        full_layers.append(np.column_stack((layer_coords, np.full(n_points, z))))    
    
    vertices = np.vstack(full_layers)
    faces = np.array(faces)
    return vertices, faces
        



