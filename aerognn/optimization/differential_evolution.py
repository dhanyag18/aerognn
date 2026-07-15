import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
from aerognn.models.uncertainty import mc_dropout_predict
from aerognn.optimization.search import greedy_diverse


FEATURE_NAMES = ['n', 'm', 'AR', 'twist', 'bulge', 'taper','setbacks', 'setback_ratio', 'chamfer']

SAFE_RANGES = {
    'n':(0.5, 12),
    'm': (1, 12),
    'AR': (0.5, 8),
    'twist': (-360, 360),
    'bulge': (0.5, 2),
    'taper': (0.5, 2),
    'setbacks':(0, 3),
    'setback_ratio': (0.05, 0.65),
    'chamfer': (0, 15),
}

INT_FEATURES = {'m', 'setbacks'}


class ShapeOptimizer:
    def __init__(self, model, pipeline, confidence_threshold, penalty=1.0):
        self.model = model
        self.pipeline = pipeline
        self.confidence_threshold = confidence_threshold
        self.penalty = penalty
        self.eval_log = []  

    def _clean_params(self, raw_params):
        params = []
        for name, val in zip(FEATURE_NAMES, raw_params):
            lo, hi = SAFE_RANGES[name]
            if name in INT_FEATURES:
                val = int(round(val))
                val = max(int(lo), min(int(hi), val))
            else:
                val = float(np.clip(val, lo, hi))
            params.append(val)
        return params

    def evaluate_design(self, params):
        graph = self.pipeline.params_to_graph(*params)
        result = mc_dropout_predict(self.model, graph, n_samples=20)

        score = result['score_mean']
        uncertainty = result['score_std']

        if uncertainty > self.confidence_threshold:
            return score + self.penalty

        return score

    def objective(self, raw_params):
        params = self._clean_params(raw_params)
        score = self.evaluate_design(params)
        self.eval_log.append((params, score))
        return score

    def run(self, maxiter=10, popsize=10, seed=42):
        bounds = [SAFE_RANGES[f] for f in FEATURE_NAMES]
        result = differential_evolution(
            self.objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            polish=True,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            updating='deferred',
            workers=1,
        )
        return result

    def top_diverse_designs(self, n=10, pool_size=100):
        sorted_log = sorted(self.eval_log, key=lambda t: t[1])
        pool_size = min(pool_size, len(sorted_log))
        pool = sorted_log[:pool_size]

        df = pd.DataFrame([dict(zip(FEATURE_NAMES, p)) for p, s in pool])
        df['score'] = [s for _, s in pool]

        scaler = StandardScaler()
        scaler.fit(df[FEATURE_NAMES])

        n = min(n, len(df))
        selected = greedy_diverse(df, 'score', n, scaler, FEATURE_NAMES, ascending=True)

        return [
            (row[FEATURE_NAMES].to_dict(), float(row['score']))
            for _, row in selected.iterrows()
        ]