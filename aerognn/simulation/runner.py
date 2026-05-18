import subprocess
import time

class SimulationRunner:

    def __init__(self, container_name='openfoam_daemon',
                 host_base_path=None,
                 container_base_path='/home/openfoam'):
        self.container = container_name
        self.host_base = host_base_path
        self.container_base = container_base_path

    def _run_cmd(self, cmd, host_case_path):
        print(f'  >> {cmd}', flush=True)
        container_case_path = host_case_path.replace(
            self.host_base, self.container_base
        )
        full_cmd = (
            f'source /usr/lib/openfoam/openfoam2506/etc/bashrc && '
            f'cd {container_case_path} && {cmd}'
        )
        subprocess.run(
            ['docker', 'exec', self.container, 'bash', '-c', full_cmd],
            check=True
        )
    def run_case(self, case_path: str, resolution: dict) -> dict:
        start_time = time.time()
        try:
            self._run_cmd('surfaceFeatureExtract', case_path)
            self._run_cmd('blockMesh', case_path)
            self._run_cmd('snappyHexMesh -overwrite', case_path)
            self._run_cmd(
                'postProcess -func writeCellCentres -constant -time 0',
                case_path
            )
            self._run_cmd('decomposePar', case_path)
            self._run_cmd('mpirun --allow-run-as-root -np 4 pimpleFoam -parallel', case_path)
            self._run_cmd('reconstructPar', case_path)
            elapsed = time.time() - start_time
            return {
                'status': 'success',
                'elapsed_minutes': elapsed / 60,
                'case_path': case_path,
            }
        except subprocess.CalledProcessError as e:
            print(f'FAILED {e}', flush=True)
            return {
                'status': 'failed',
                'error': str(e),
                'case_path': case_path,
            }

    def run_batch(self, case_paths: list, resolution: dict) -> list:
        results = []
        for i, path in enumerate(case_paths):
            print(f'Running case {i+1}/{len(case_paths)}: {path}', flush=True)
            result = self.run_case(path, resolution)
            results.append(result)
            print(
                f'  Status: {result["status"]}, '
                f'Time: {result.get("elapsed_minutes", 0):.1f} min',
                flush=True
            )
        return results