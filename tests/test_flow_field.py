import numpy as np
import torch
from aerognn.data.flow_field_loader import load_openfoam_field
from aerognn.data.volumetric_graph import build_volumetric_graph, classify_nodes


CASE_DIR = "/Users/dhanyaganesh/Downloads/aerognn/data/raw/1"


def make_fields():
    cell_centers = np.array([
        [0.5,0.5, 0.5],
        [1.5, 0.5, 0.5],
        [0.5, 1.5, 0.5],
        [1.5, 1.5, 0.5],
        [0.5, 0.5, 1.5],
        [1.5, 0.5, 1.5],
    ], dtype=np.float32)

    cell_neighbors = [
        [1,2],
        [0,3],
        [0,3],
        [1,2],
        [5],
        [4],
    ]

    boundary_info = {
        0: 'inlet',
        1: 'outlet',
        2: 'wall',
    }

    velocity = np.array([
        [5.0, 0.0, 0.0],
        [4.8, 0.1, 0.0],
        [0.0, 0.0, 0.0],
        [3.0, 0.2, 0.0],
        [6.0, 0.0, 0.0],
        [5.5, 0.1, 0.0],
    ], dtype=np.float32)
    
    pressure = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    return cell_centers, cell_neighbors, boundary_info, velocity, pressure

def test_node_count():
    cell_centers, cell_neighbors, boundary_info, velocity, pressure = make_fields()
    data = build_volumetric_graph(
        cell_centers=cell_centers,
        cell_neighbors=cell_neighbors,
        boundary_info=boundary_info,
        velocity=velocity,
        pressure=pressure,
        id=1,
        cd_mean=0.5,
        cl_mean=0.1,
        cl_std=0.05,
    )
    assert data.num_nodes == 6
def test_feature_dim():
    cell_centers, cell_neighbors, boundary_info, velocity, pressure = make_fields()
    data = build_volumetric_graph(
        cell_centers=cell_centers,
        cell_neighbors=cell_neighbors,
        boundary_info=boundary_info,
        velocity=velocity,
        pressure=pressure,
        id=1,
        cd_mean=0.5,
        cl_mean=0.1,
        cl_std=0.05,
    )
    assert data.x.shape == (6,12)
def test_edge_index_valid():
    cell_centers, cell_neighbors, boundary_info, velocity, pressure = make_fields()
    data = build_volumetric_graph(
        cell_centers=cell_centers,
        cell_neighbors=cell_neighbors,
        boundary_info=boundary_info,
        velocity=velocity,
        pressure=pressure,
        id=1,
        cd_mean=0.5,
        cl_mean=0.1,
        cl_std=0.05,
    )
    assert data.edge_index.min() >= 0
    assert data.edge_index.max() < data.num_nodes

def test_edges_bidirectional():
    cell_centers, cell_neighbors, boundary_info, velocity, pressure = make_fields()
    data = build_volumetric_graph(
        cell_centers=cell_centers,
        cell_neighbors=cell_neighbors,
        boundary_info=boundary_info,
        velocity=velocity,
        pressure=pressure,
        id=1,
        cd_mean=0.5,
        cl_mean=0.1,
        cl_std=0.05,
    )
    edges = set(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    for (i,j) in edges:
        assert (j,i) in edges

def test_y_scores():
    cell_centers, cell_neighbors, boundary_info, velocity, pressure = make_fields()
    data = build_volumetric_graph(
        cell_centers=cell_centers,
        cell_neighbors=cell_neighbors,
        boundary_info=boundary_info,
        velocity=velocity,
        pressure=pressure,
        id=1,
        cd_mean=0.5,
        cl_mean=0.1,
        cl_std=0.05,
    )
    actual = torch.tensor([[0.5, 0.1, 0.05]], dtype=torch.float)
    assert data.y.shape == (1,3)
    assert torch.allclose(data.y, actual)

def test_wall_node_zero_velocity():
    cell_centers, cell_neighbors, boundary_info, velocity, pressure = make_fields()
    velocity[2] = [0.0, 0.0, 0.0]
    data = build_volumetric_graph(
        cell_centers=cell_centers,
        cell_neighbors=cell_neighbors,
        boundary_info=boundary_info,
        velocity=velocity,
        pressure=pressure,
        id=1,
        cd_mean=0.5,
        cl_mean=0.1,
        cl_std=0.05,
    )
    wall_vel = data.y_velocity[2]
    assert torch.allclose(wall_vel, torch.zeros(3), atol=1e-6)

def test_node_type_classification():
    cell_centers, cell_neighbors, boundary_info, velocity, pressure = make_fields()
    data = build_volumetric_graph(
        cell_centers=cell_centers,
        cell_neighbors=cell_neighbors,
        boundary_info=boundary_info,
        velocity=velocity,
        pressure=pressure,
        id=1,
        cd_mean=0.5,
        cl_mean=0.1,
        cl_std=0.05,
    )
    assert data.node_types[0].item() == 2
    assert data.node_types[1].item() == 3
    assert data.node_types[2].item() == 1
    assert data.node_types[3].item() == 0
    assert data.node_types[4].item() == 0
    assert data.node_types[5].item() == 0

def test_parser_velocity_shape():
    fields = load_openfoam_field(CASE_DIR)
    assert fields['velocity'].ndim==2
    assert fields['velocity'].shape[1] == 3

def test_parser_pressure_shape():
    fields = load_openfoam_field(CASE_DIR)
    assert fields['pressure'].ndim == 1

def test_parser_fields_consistent():
    fields = load_openfoam_field(CASE_DIR)
    n = fields['cell_centers'].shape[0]
    assert fields['velocity'].shape[0]==n
    assert fields['pressure'].shape[0]==n

def test_parser_known_cell_velocity():
    fields = load_openfoam_field(CASE_DIR)
    expected = np.array([6.284658053, -0.000866510787, 0.0008165433528], dtype=np.float32)
    np.testing.assert_allclose(fields['velocity'][0], expected, atol=1e-4)

def test_inlet_nodes_classified():
    fields = load_openfoam_field(CASE_DIR)
    inlet_indices = [i for i, t in fields['boundary_info'].items() if t == 'inlet']
    node_types = classify_nodes(fields['cell_centers'], fields['boundary_info'])
    for i in inlet_indices:
        assert node_types[i].item() == 2

def test_outlet_nodes_classified():
    fields = load_openfoam_field(CASE_DIR)
    outlet_indices = [i for i, t in fields['boundary_info'].items() if t=='outlet']
    node_types = classify_nodes(fields['cell_centers'], fields['boundary_info'])
    for i in outlet_indices:
        assert node_types[i].item()==3