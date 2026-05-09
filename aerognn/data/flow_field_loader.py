import numpy as np
import os

def _parse_vector_field(path):
    vectors = []
    with open(path, 'r') as f:
        lines = f.readlines()
    start_idx = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.isdigit():
            start_idx = i + 2
            break
    for line in lines[start_idx:]:
        stripped = line.strip()
        if stripped == ')':
            break
        values = [float(v) for v in stripped.strip('()').split()]
        vectors.append(values)
    return np.array(vectors)

def _parse_scalar_field(path):
    scalars = []
    with open(path, 'r') as f:
        lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.isdigit():
            start_idx = i + 2
            break
    for line in lines[start_idx:]:
        stripped = line.strip()
        if stripped == ')':
            break
        scalars.append(float(stripped))
    return np.array(scalars)

def _compute_cell_centers(time_dir):
    c_path = os.path.join(time_dir, 'C')
    return _parse_vector_field(c_path)

def _compute_cell_neighbors(mesh_path):
    owner_path = os.path.join(mesh_path, 'owner')
    owners = _parse_scalar_field(owner_path).astype(int)
    neighbour_path = os.path.join(mesh_path, 'neighbour')
    neighbours = _parse_scalar_field(neighbour_path).astype(int)
    n_cells = int(max(owners.max(), neighbours.max())) + 1
    cell_neighbors = [[] for _ in range(n_cells)]
    for owner, neighbour in zip(owners, neighbours):
        cell_neighbors[owner].append(neighbour)
        cell_neighbors[neighbour].append(owner)
    return cell_neighbors

def _parse_boundary_info(mesh_path):
    boundary_path = os.path.join(mesh_path, 'boundary')
    boundary_info = {}
    with open(boundary_path, 'r') as f:
        lines = f.readlines()
    owner_path = os.path.join(mesh_path, 'owner')
    owners = _parse_scalar_field(owner_path).astype(int)
    
    current_patch = None
    n_faces = None
    start_face = None
    
    recognized = ('inlet', 'outlet', 'building', 'ground',
                  'top', 'frontAndBack', 'sky', 'sides', 'symmetry')
    
    for line in lines:
        stripped = line.strip()
        if stripped in recognized:
            current_patch = stripped
        if stripped.startswith('nFaces'):
            n_faces = int(stripped.split()[1].rstrip(';'))
        if stripped.startswith('startFace'):
            start_face = int(stripped.split()[1].rstrip(';'))
        if current_patch and n_faces is not None and start_face is not None:
            for face_id in range(start_face, start_face + n_faces):
                cell_id = owners[face_id]
                if current_patch == 'building':
                    boundary_info[cell_id] = 'wall'
                elif current_patch == 'inlet':
                    boundary_info[cell_id] = 'inlet'
                elif current_patch == 'outlet':
                    boundary_info[cell_id] = 'outlet'
                else:
                    boundary_info[cell_id] = 'symmetry'
            current_patch = None
            n_faces = None
            start_face = None
    
    return boundary_info

def _find_latest_time(case_dir: str):
    entries = os.listdir(case_dir)
    time_dirs = {}
    for entry in entries:
        try:
            time_dirs[float(entry)] = entry
        except ValueError:
            continue
    return time_dirs[max(time_dirs)]

def load_openfoam_field(case_dir: str, time_step: str = None):
    if time_step is None:
        time_step = _find_latest_time(case_dir)

    fields = {}
    u_path = os.path.join(case_dir, time_step, 'U')
    fields['velocity'] = _parse_vector_field(u_path)
    p_path = os.path.join(case_dir, time_step, 'p')
    fields['pressure'] = _parse_scalar_field(p_path)
    
    mesh_path = os.path.join(case_dir, 'constant', 'polyMesh')
    time_dir = os.path.join(case_dir, time_step)
    fields['cell_centers'] = _compute_cell_centers(time_dir)
    fields['cell_neighbors'] = _compute_cell_neighbors(mesh_path)
    fields['boundary_info'] = _parse_boundary_info(mesh_path)
   
    return fields