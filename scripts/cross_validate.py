from aerognn.data.dataset import BuildingDataset
from aerognn.training.trainer import cross_validation
def cv():
    dataset = BuildingDataset()
    mae, r2 = cross_validation(dataset, 150)
    print(f'MAE: {mae}, R^2: {r2}')

if __name__ == "__main__":
    cv()
