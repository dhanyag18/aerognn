import numpy as np
import trimesh
import os
import subprocess
import shutil
import uuid
from shapely.geometry import Polygon
from shapely.ops import orient

from aerognn.data.volumetric_graph import build_volumetric_graph, subsample_mesh
from aerognn.data.flow_field_loader import _compute_cell_centers, _compute_cell_neighbors, _parse_boundary_info, _find_latest_time

CONTAINER_NAME = 'openfoam_daemon'
HOST_TEMPLATE_CASE = '/Users/dhanyaganesh/Downloads/openfoam/vortex'
HOST_AEROGNN = '/Users/dhanyaganesh/Downloads/aerognn'
CONTAINER_HOME = '/home/openfoam'


def _run_openfoam_cmd(cmd, host_case_path):
    container_case_path = host_case_path.replace(HOST_AEROGNN, CONTAINER_HOME)
    full_cmd = f'source /usr/lib/openfoam/openfoam2506/etc/bashrc && cd {container_case_path} && {cmd}'
    subprocess.run(
        ['docker', 'exec', CONTAINER_NAME, 'bash', '-c', full_cmd],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def _superformula_radius(theta, m, n1, n2, n3, a=1, b=1):
    term1 = np.power(np.abs(np.cos(m * theta / 4) / a), n2)
    term2 = np.power(np.abs(np.sin(m * theta / 4) / b), n3)
    return np.power(term1 + term2, -1 / n1)


def _apply_chamfer(coords, dist):
    if dist == 0:
        return coords
    norm = np.linalg.norm(coords, axis=1)
    max_r = np.max(norm)
    threshold = max_r - dist
    scale_factors = np.where(norm > threshold, threshold / norm, 1.0)
    return coords * scale_factors[:, np.newaxis]


def _build_base_cross_section(m, n, ar, num_points, target_area=5000.0):
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    r = _superformula_radius(theta, m, n, n, n)
    x = r * np.cos(theta) * ar
    y = r * np.sin(theta)
    coords = np.column_stack([x, y])
    poly = Polygon(coords).buffer(0)
    scale = np.sqrt(target_area / poly.area)
    final_coords = np.array(poly.exterior.coords) * scale
    poly = orient(Polygon(final_coords), sign=1.0)
    boundary = poly.exterior
    distances = np.linspace(0, boundary.length, num_points, endpoint=False)
    base_coords = np.array(
        [(boundary.interpolate(d).x, boundary.interpolate(d).y) for d in distances]
    )
    base_coords -= np.mean(base_coords, axis=0)
    return base_coords


def generate_building_stl(n, m, ar, twist, bulge, taper, setbacks, setback_ratio, chamfer, 
                          output_path, num_points=240, num_layers=200, height=200.0):
    
    base_coords = _build_base_cross_section(m, n, ar, num_points)
    layers = []
    faces = []
    widths = []
    for i in range(num_layers + 1):
        frac = i / num_layers
        z = frac * height

        tier = min(int(np.floor(frac * (setbacks + 1))), setbacks)
        s_scale = (1.0 - setback_ratio) ** tier
        t_scale = 1.0 + (taper - 1.0) * frac
        b_scale = 1.0 + (bulge - 1.0) * np.sin(np.pi * frac)

        angle = np.radians(frac * twist)
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])

        transformed = (base_coords @ rot.T) * (t_scale * s_scale * b_scale)
        transformed = _apply_chamfer(transformed, chamfer)

        layers.append(np.column_stack([transformed, np.full(num_points, z)]))
        widths.append(np.ptp(transformed[:, 0]))

        if i > 0:
            for j in range(num_points):
                v1 = (i - 1) * num_points + j
                v2 = (i - 1) * num_points + (j + 1) % num_points
                v3 = i * num_points + j
                v4 = i * num_points + (j + 1) % num_points
                faces.append([v2, v1, v3])
                faces.append([v4, v2, v3])

    vertices = np.vstack(layers)

    bot_idx = np.arange(num_points)
    top_idx = np.arange(len(vertices) - num_points, len(vertices))
    for i in range(1, num_points - 1):
        faces.append([bot_idx[0], bot_idx[i + 1], bot_idx[i]])
        faces.append([top_idx[0], top_idx[i], top_idx[i + 1]])

    widths = np.array(widths)
    aref = float(np.trapezoid(widths, dx=(height / num_layers)))
    lref = float(np.max(widths))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    mesh.fix_normals()
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)

    if not mesh.is_watertight:
        raise RuntimeError(f"Generated STL is not watertight: {output_path}")

    mesh.export(output_path, file_type='stl')

    return output_path, aref, lref


def run_snappy(stl_path, case_id='temp', output_dir='temp_cases'):
    os.makedirs(output_dir, exist_ok=True)
    case_path = os.path.join(output_dir, f'case_{case_id}')

    if os.path.exists(case_path):
        shutil.rmtree(case_path)
    shutil.copytree(HOST_TEMPLATE_CASE, case_path)

    stl_dest = os.path.join(case_path, 'constant', 'triSurface', 'building.stl')
    os.makedirs(os.path.dirname(stl_dest), exist_ok=True)
    shutil.copy(stl_path, stl_dest)

    _run_openfoam_cmd('surfaceFeatureExtract', case_path)
    _run_openfoam_cmd('blockMesh', case_path)
    _run_openfoam_cmd('snappyHexMesh -overwrite', case_path)
    _run_openfoam_cmd('postProcess -func writeCellCentres -constant -time 0', case_path)

    time_step = _find_latest_time(case_path)
    time_dir = os.path.join(case_path, time_step)

    mesh_path = os.path.join(case_path, 'constant', 'polyMesh')
    cell_centers = _compute_cell_centers(time_dir)
    cell_neighbors = _compute_cell_neighbors(mesh_path)
    boundary_info = _parse_boundary_info(mesh_path)

    shutil.rmtree(case_path)

    return cell_centers, cell_neighbors, boundary_info


def params_to_graph(n, m, ar, twist, bulge, taper, setbacks, setback_ratio, chamfer):
    call_id = uuid.uuid4().hex[:8]

    stl_path = os.path.join(HOST_AEROGNN, f'temp_building_{call_id}.stl')
    _, aref, lref = generate_building_stl(n, m, ar, twist, bulge, taper,
                                           setbacks, setback_ratio, chamfer, stl_path)

    output_dir = os.path.join(HOST_AEROGNN, 'temp_cases')
    cell_centers, cell_neighbors, boundary_info = run_snappy(
        stl_path, case_id=call_id, output_dir=output_dir)

    n_cells = len(cell_centers)
    velocity = np.zeros((n_cells, 3))
    pressure = np.zeros(n_cells)

    final_cell_centers, final_cell_neighbors, final_boundary_info, final_velocity, final_pressure = subsample_mesh(
        cell_centers, cell_neighbors, boundary_info,
        velocity, pressure,
        max_wall_nodes=3000, max_interior_nodes=2000
    )

    graph = build_volumetric_graph(
        cell_centers=final_cell_centers,
        cell_neighbors=final_cell_neighbors,
        boundary_info=final_boundary_info,
        velocity=final_velocity,
        pressure=final_pressure,
        id = 0,
        cd_mean=0.0,
        cl_mean=0.0,
        cl_std=0.0
    )

    if os.path.exists(stl_path):
        os.remove(stl_path)

    return graph