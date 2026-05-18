from aerognn.data.flow_field_loader import (
    load_openfoam_field,
    parse_force_coefficients
)
from aerognn.data.volumetric_graph import build_volumetric_graph, subsample_mesh

def extract_simulation_result(
    case_dir: str,
    case_id: int = 0,
    max_wall_nodes: int = 3000,
    max_interior_nodes: int = 2000,
):
    fields = load_openfoam_field(case_dir)
    coeffs = parse_force_coefficients(case_dir)

    centers, neighbors, boundary, vel, pres = subsample_mesh(
        fields['cell_centers'],
        fields['cell_neighbors'],
        fields['boundary_info'],
        fields['velocity'],
        fields['pressure'],
        max_wall_nodes=max_wall_nodes,
        max_interior_nodes=max_interior_nodes,
    )

    graph = build_volumetric_graph(
        cell_centers=centers,
        cell_neighbors=neighbors,
        boundary_info=boundary,
        velocity=vel,
        pressure=pres,
        id=case_id,
        cd_mean=coeffs['cd_mean'],
        cl_mean=coeffs['cl_mean'],
        cl_std=coeffs['cl_std'],
    )

    return graph