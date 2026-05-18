import os
import re
import shutil
import json
from aerognn.data.params_to_graph import generate_building_stl

class ResolutionConfig:
    COARSE = {
        'name': 'coarse',
        'block_cells': [20, 20, 20],
        'refinement_levels': 1,
        'boundary_layers': 0,
        'end_time': 150,
        'write_interval': 50,
        'estimated_cells': 10000,
        'estimated_time_minutes': 30,
    }
    FINE = {
        'name': 'fine',
        'block_cells': [40, 25, 40],
        'refinement_levels': 3,
        'boundary_layers': 3,
        'end_time': 250,
        'write_interval': 125,
        'estimated_cells': 120000,
        'estimated_time_minutes': 120,
    }

def _update_force_coeffs(case_path, aref, lref):
    cd_path = os.path.join(case_path, 'system', 'controlDict')
    with open(cd_path, 'r') as f:
        lines = f.readlines()
    with open(cd_path, 'w') as f:
        for line in lines:
            if 'Aref' in line and 'lRef' not in line:
                f.write(f'        Aref            {aref:.4f};\n')
            elif 'lRef' in line:
                f.write(f'        lRef            {lref:.4f};\n')
            else:
                f.write(line)

def _update_block_mesh_dict(case_path, block_cells):
    bmd_path = os.path.join(case_path, 'system', 'blockMeshDict')
    with open(bmd_path, 'r') as f:
        content = f.read()
    cx, cy, cz = block_cells
    content = re.sub(
        r'hex \(0 1 2 3 4 5 6 7\) \(\d+ \d+ \d+\)',
        f'hex (0 1 2 3 4 5 6 7) ({cx} {cy} {cz})',
        content
    )
    with open(bmd_path, 'w') as f:
        f.write(content)

def _update_snappy_hex_mesh_dict(case_path, resolution):
    shmd_path = os.path.join(case_path, 'system', 'snappyHexMeshDict')
    if not os.path.exists(shmd_path):
        return
    with open(shmd_path, 'r') as f:
        content = f.read()

    ref_levels = resolution['refinement_levels']
    content = re.sub(
        r'(level\s*\()(\d+)\s+(\d+)(\s*\))',
        lambda m: f"{m.group(1)}{ref_levels} {ref_levels}{m.group(4)}",
        content
    )

    n_layers = resolution['boundary_layers']
    content = re.sub(
        r'(nSurfaceLayers\s+)\d+',
        f'nSurfaceLayers {n_layers}',
        content
    )

    with open(shmd_path, 'w') as f:
        f.write(content)

def _update_control_dict(case_path, resolution):
    cd_path = os.path.join(case_path, 'system', 'controlDict')
    with open(cd_path, 'r') as f:
        content = f.read()

    content = re.sub(
        r'^(endTime\s+)\S+;',
        f'endTime         {resolution["end_time"]};',
        content,
        flags=re.MULTILINE
    )

    if 'functions' in content:
        top, bottom = content.split('functions', 1)
        top = re.sub(
            r'^(writeInterval\s+)\S+;',
            f'writeInterval   {resolution["write_interval"]};',
            top,
            flags=re.MULTILINE
        )
        content = top + 'functions' + bottom
    else:
        content = re.sub(
            r'^(writeInterval\s+)\S+;',
            f'writeInterval   {resolution["write_interval"]};',
            content,
            flags=re.MULTILINE
        )

    with open(cd_path, 'w') as f:
        f.write(content)

def generate_case(
    params: dict,
    case_id: str,
    resolution: dict,
    template_dir: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    case_path = os.path.join(output_dir, f'case_{case_id}')

    if os.path.exists(case_path):
        shutil.rmtree(case_path)
    shutil.copytree(template_dir, case_path)

    stl_path = os.path.join(case_path, 'constant', 'triSurface', 'building.stl')
    os.makedirs(os.path.dirname(stl_path), exist_ok=True)
    _, aref, lref = generate_building_stl(
        params['n'], params['m'], params['AR'],
        params['twist'], params['bulge'], params['taper'],
        params['setbacks'], params['setback_ratio'],
        params['chamfer'], stl_path
    )
    _update_force_coeffs(case_path, aref, lref)
    _update_block_mesh_dict(case_path, resolution['block_cells'])
    _update_snappy_hex_mesh_dict(case_path, resolution)
    _update_control_dict(case_path, resolution)
    metadata = {
        'params': params,
        'resolution': resolution['name'],
        'case_id': case_id,
        'aref': aref,
        'lref': lref,
    }
    with open(os.path.join(case_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return case_path