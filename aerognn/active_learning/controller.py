from aerognn.simulation.case_generator import ResolutionConfig
from aerognn.data.params_to_graph import params_to_graph
import numpy as np
import pandas as pd
import csv
import os


class ActiveLearningController:

    def __init__(self, model, generate_case_fn, template_dir, output_dir,
                 runner, extract_fn, dataset, trainer, acquisition):
        self.model = model
        self.generate_case = generate_case_fn
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.runner = runner
        self.extract = extract_fn
        self.dataset = dataset
        self.trainer = trainer
        self.acquisition = acquisition

    def _generate_random_candidates(self, n_candidates):
        SAFE_RANGES = {
            'n': (0.5, 12),
            'm': (1, 12),
            'AR': (0.5, 8),
            'twist': (-360, 360),
            'bulge': (0.5, 2),
            'taper': (0.5, 2),
            'setbacks': (0, 3),
            'setback_ratio': (0.05, 0.65),
            'chamfer': (0, 15)
        }
        INT_FEATURES = {'m', 'setbacks'}

        params_df = pd.DataFrame()
        for f, (lo, hi) in SAFE_RANGES.items():
            vals = np.random.uniform(lo, hi, n_candidates)
            if f in INT_FEATURES:
                vals = np.round(vals).astype(int).clip(int(lo), int(hi))
            params_df[f] = vals

        graphs = []
        for _, row in params_df.iterrows():
            graph = params_to_graph(
                row['n'], int(row['m']), row['AR'],
                row['twist'], row['bulge'], row['taper'],
                int(row['setbacks']), row['setback_ratio'],
                row['chamfer']
            )
            graphs.append(graph)

        return graphs, params_df

    def _log_selected(self, selected_params, param_cols, next_id, iteration):
        log_path = 'data/coarse/active_learning_log.csv'
        file_exists = os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['iteration', 'case_id'] + param_cols)
            if not file_exists:
                writer.writeheader()
            for i, params in enumerate(
                    selected_params[param_cols].to_dict('records')):
                writer.writerow({
                    'iteration': iteration,
                    'case_id': next_id + i,
                    **params
                })

    def _log_times(self, results, next_id, iteration):
        time_log_path = 'data/coarse/simulation_times.csv'
        file_exists = os.path.exists(time_log_path)
        with open(time_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['iteration', 'case_id', 'status', 'elapsed_minutes'])
            if not file_exists:
                writer.writeheader()
            for i, result in enumerate(results):
                writer.writerow({
                    'iteration': iteration,
                    'case_id': next_id + i,
                    'status': result['status'],
                    'elapsed_minutes': round(result.get('elapsed_minutes', 0), 2)
                })

    def run_iteration(self, n_designs, resolution, n_random_candidates, iteration):
        
        FEATURE_NAMES = ['n', 'm', 'AR', 'twist', 'bulge', 'taper','setbacks', 'setback_ratio', 'chamfer']
        PARAM_COLS = FEATURE_NAMES

        existing_ids = [d.id for d in self.dataset.coarse]
        next_id = max(existing_ids) + 1

        candidates, candidates_df = self._generate_random_candidates(n_random_candidates)

        selected_graphs, selected_params = self.acquisition.select_batch(candidates, candidates_df, n_designs, FEATURE_NAMES)

        self._log_selected(selected_params, PARAM_COLS, next_id, iteration)

        res_config = (ResolutionConfig.COARSE
                      if resolution == 'coarse'
                      else ResolutionConfig.FINE)

        case_paths = []
        for i, params in enumerate(selected_params[PARAM_COLS].to_dict('records')):
            path = self.generate_case(
                params=params,
                case_id=f'al_{next_id + i}',
                resolution=res_config,
                template_dir=self.template_dir,
                output_dir=self.output_dir
            )
            case_paths.append(path)

        results = self.runner.run_batch(case_paths, res_config)

        n_success = 0
        for i, result in enumerate(results):
            if result['status'] == 'success':
                case_id = next_id + i
                graph = self.extract(result['case_path'], case_id=case_id)
                self.dataset.add_simulation(graph, resolution)
                n_success += 1
        print(f'{n_success}/{n_designs} simulations succeeded')

        self._log_times(results, next_id, iteration)

        print(f'Retraining on {len(list(self.dataset.coarse))} samples...')
        self.trainer.retrain(self.model, self.dataset)

        return n_success

    def run(self, n_iterations=10, n_designs_per_iter=5, resolution='coarse', n_random_candidates=200):
        for i in range(n_iterations):
            print(f'\n*** Iteration {i+1}/{n_iterations} ***')
            n_success = self.run_iteration(
                n_designs=n_designs_per_iter,
                resolution=resolution,
                n_random_candidates=n_random_candidates,
                iteration=i + 1
            )
            if n_success == 0:
                print('No successful simulations. Stopping.')
                break