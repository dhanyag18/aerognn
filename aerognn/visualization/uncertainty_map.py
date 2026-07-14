import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors


def plot_uncertainty_map(wall_pos, pressure_std, verts, faces, title='Prediction Uncertainty', output_path=None):
    
    nbrs = NearestNeighbors(n_neighbors=1).fit(wall_pos)
    _, indices = nbrs.kneighbors(verts)
    uncertainty = pressure_std[indices.squeeze()]

    pv_faces = np.hstack([np.full((len(faces), 1), 3), faces])
    mesh = pv.PolyData(verts.astype(np.float32), pv_faces)
    mesh['uncertainty'] = uncertainty.astype(np.float32)

    plotter = pv.Plotter(off_screen=output_path is not None)
    plotter.add_mesh(mesh, scalars='uncertainty', cmap='RdBu_r', scalar_bar_args={'title': 'Uncertainty (std)'})
    plotter.camera_position = [
        (-250, -250, 250),
        (0, 0, 100),
        (0, 0, 1)
    ]
    plotter.reset_camera()

    if output_path:
        plotter.screenshot(output_path)
    else:
        plotter.show()