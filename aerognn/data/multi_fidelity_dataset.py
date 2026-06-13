import torch
from aerognn.data.dataset import BuildingDataset

class MultiFidelityDataset:
    def __init__(self, coarse_dir='data/coarse/',
                 fine_dir='data/processed/'):
        self.coarse_dir = coarse_dir
        self.fine_dir = fine_dir

    @property
    def coarse(self):
        return BuildingDataset(self.coarse_dir)

    @property
    def fine(self):
        return BuildingDataset(self.fine_dir)

    def get_paired(self):
        coarse_ids = {d.id for d in self.coarse}
        fine_ids = {d.id for d in self.fine}
        paired_ids = coarse_ids & fine_ids
        return (
            [d for d in self.coarse if d.id in paired_ids],
            [d for d in self.fine   if d.id in paired_ids]
        )

    def add_simulation(self, graph, resolution):
        if resolution == 'coarse':
            path = f'data/coarse/{graph.id}.pt'
        else:
            path = f'data/processed/{graph.id}.pt'
        torch.save(graph, path)