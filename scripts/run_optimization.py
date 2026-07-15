import os
import json
import torch
import numpy as np
from aerognn.models.mutli_fidelity import MultiFidelityPINNGNN
from aerognn.models.uncertainty import mc_dropout_predict
from aerognn.data.multi_fidelity_dataset import MultiFidelityDataset
import aerognn.data.params_to_graph as pipeline
from aerognn.optimization.differential_evolution import ShapeOptimizer  

HOST_AEROGNN = '/Users/dhanyaganesh/Downloads/aerognn'
MODEL_PATH = f'{HOST_AEROGNN}/final_mf_model.pt'


def estimate_confidence_threshold(model, dataset, n_samples=100, n_mc=20, seed=42):
    rng = np.random.RandomState(seed)
    coarse_data = list(dataset.coarse)
    idx = rng.choice(len(coarse_data),
                      size=min(n_samples, len(coarse_data)),
                      replace=False)

    uncertainties = []
    for i in idx:
        graph = coarse_data[i]
        result = mc_dropout_predict(model, graph, n_samples=n_mc)
        uncertainties.append(float(result['score_std']))

    uncertainties = np.array(uncertainties)
    mean_u = uncertainties.mean()
    std_u = uncertainties.std()
    threshold = mean_u + 2 * std_u

    print(f"mean = {mean_u:.4f}")
    print(f"std = {std_u:.4f}")
    print(f"min = {uncertainties.min():.4f}")
    print(f"max = {uncertainties.max():.4f}")
    print(f"confidence_threshold (mean + 2*std) = {threshold:.4f}")

    return threshold

def run_optimization():
    model = MultiFidelityPINNGNN()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    dataset = MultiFidelityDataset()

    confidence_threshold = estimate_confidence_threshold(model, dataset)

    optimizer = ShapeOptimizer(model, pipeline, confidence_threshold)
    result = optimizer.run(maxiter=10, popsize=10, seed=42)

    print(f"\nDE finished. Best score: {result.fun:.4f}")
    print(f"Total evaluations: {len(optimizer.eval_log)}")

    top10 = optimizer.top_diverse_designs(n=10, pool_size=100)

    for i, (params, score) in enumerate(top10, 1):
        print(f"\nRank {i} | score={score:.4f}")
        for k, v in params.items():
            print(f"    {k:>14s} = {v}")

    out_path = os.path.join(HOST_AEROGNN, 'optimization_top10.json')
    with open(out_path, 'w') as f:
        json.dump([
            {'rank': i + 1, 'params': p, 'predicted_score': s}
            for i, (p, s) in enumerate(top10)
        ], f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    run_optimization()