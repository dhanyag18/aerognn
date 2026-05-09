import matplotlib.pyplot as plt
import numpy as np

def plot_training_diagnostics(loss_history):

    active = {k: v for k, v in loss_history.items() if len(v) > 0}
    
    n_plots = len(active)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, active.items()):
        values = np.array(values)
        ax.semilogy(values, linewidth=0.8, alpha=0.8)
        ax.set_title(f'{name} loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss (log scale)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig