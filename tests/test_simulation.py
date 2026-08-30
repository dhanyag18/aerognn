from aerognn.simulation.case_generator import generate_case, ResolutionConfig
from aerognn.simulation.result_extractor import extract_simulation_result
from aerognn.simulation.runner import SimulationRunner
import os

HOST_AEROGNN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(os.path.dirname(HOST_AEROGNN), 'openfoam', 'vortex')

def test_full_pipeline():
    params = {
        'n': 5.0, 'm': 4, 'AR': 2.0,
        'twist': 0, 'bulge': 1.0, 'taper': 1.0,
        'setbacks': 1, 'setback_ratio': 0.3, 'chamfer': 5.0
    }

    case_path = generate_case(params, 'test_001', ResolutionConfig.COARSE,template_dir=TEMPLATE, output_dir=os.path.join(HOST_AEROGNN, 'test_cases'))
    runner = SimulationRunner(container_name='openfoam_daemon',host_base_path=HOST_AEROGNN,container_base_path='/home/openfoam')
    result = runner.run_case(case_path, ResolutionConfig.COARSE)
    assert result['status'] == 'success'

    graph = extract_simulation_result(case_path)
    assert graph.x.shape[1] == 14
    assert graph.y_velocity.shape[1] == 3
    assert graph.y_pressure.shape[0] > 0