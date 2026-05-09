import numpy as np
import pyvista as pv


def plot_streamlines(cell_centers, velocity_field, building_mesh, output_path=None):
    cloud = pv.PolyData(cell_centers.astype(np.float32))
    cloud['velocity'] = velocity_field.astype(np.float32)
    cloud['velocity_mag'] = np.linalg.norm(velocity_field, axis=1).astype(np.float32)

    bounds = cloud.bounds
    grid = pv.ImageData()
    grid.dimensions = [20, 20, 20]
    grid.origin = bounds[::2]
    grid.spacing = [
        (bounds[1]-bounds[0])/19,
        (bounds[3]-bounds[2])/19,
        (bounds[5]-bounds[4])/19,
    ]
    interpolated = grid.interpolate(cloud, radius=15.0)

    source_x = bounds[0] + 2
    source_y = (bounds[2]+bounds[3])/2
    source_z = (bounds[4]+bounds[5])/2

    streamlines = interpolated.streamlines(
        source_center=(source_x, source_y, source_z),
        source_radius=15,
        n_points=100,
        vectors='velocity',
        max_steps=500
    )

    tubes = streamlines.tube(radius=0.3)

    plotter = pv.Plotter(off_screen=output_path is not None)
    plotter.add_mesh(building_mesh, color='gray', opacity=0.5)

    if tubes.n_points > 0:
        plotter.add_mesh(tubes, scalars='velocity_mag', cmap='jet')
    else:
        print("Warning: no streamlines generated")

    if output_path:
        plotter.screenshot(output_path)
    else:
        plotter.show()
