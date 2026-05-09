import numpy as np
import pyvista as pv
import os
import subprocess
import shutil

from aerognn.geometry.superformula import generate_cross_section, apply_aspect_ratio, normalize_area
from aerognn.geometry.extrusion import extrude_building
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


def generate_building_stl(n, m, ar, twist, bulge, taper,
                           setbacks, setback_ratio, chamfer,
                           output_path, num_points=100, num_layers=50):
    cross_section = generate_cross_section(n, n, n, m, num_points)
    cross_section = apply_aspect_ratio(cross_section, ar)
    cross_section = normalize_area(cross_section, target_area=5000.0)
    vertices, faces = extrude_building(
        num_layers=num_layers,
        taper=taper,
        bulge=bulge,
        helical_twist=twist,
        num_setbacks=setbacks,
        setback_reduction=setback_ratio,
        chamfer_distance=chamfer,
        cross_section=cross_section
    )
    pv_faces = np.hstack([np.full((len(faces), 1), 3), faces])
    mesh = pv.PolyData(vertices, pv_faces)
    mesh = mesh.clean()
    mesh = mesh.fill_holes(100)
    mesh.save(output_path)
    return output_path


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
    print("Dirs in case:", os.listdir(case_path))
    time_step = _find_latest_time(case_path)
    print("Time step found:", time_step)
    time_dir = os.path.join(case_path, time_step)
    print("Time dir contents:", os.listdir(time_dir) if os.path.exists(time_dir) else "NOT FOUND")

    cell_centers = _compute_cell_centers(time_dir)
    cell_neighbors = _compute_cell_neighbors(mesh_path)
    boundary_info = _parse_boundary_info(mesh_path)

    shutil.rmtree(case_path)

    return cell_centers, cell_neighbors, boundary_info


def params_to_graph(n, m, ar, twist, bulge, taper, setbacks, setback_ratio, chamfer):
    stl_path = os.path.join(HOST_AEROGNN, 'temp_building.stl')
    generate_building_stl(n, m, ar, twist, bulge, taper,
                          setbacks, setback_ratio, chamfer, stl_path)

    output_dir = os.path.join(HOST_AEROGNN, 'temp_cases')
    cell_centers, cell_neighbors, boundary_info = run_snappy(stl_path, output_dir=output_dir)

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
        id=0,
        cd_mean=0.0,
        cl_mean=0.0,
        cl_std=0.0
    )

    if os.path.exists(stl_path):
        os.remove(stl_path)

    return graph