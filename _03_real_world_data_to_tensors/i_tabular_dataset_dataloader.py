import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WineDataset(Dataset):
    def __init__(self):
        wine_path = os.path.join(os.path.pardir, "_00_data", "d_tabular-wine", "winequality-white.csv")
        wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
        wineq = torch.from_numpy(wineq_numpy)

        data = wineq[:, :-1]  # Selects all rows and all columns except the last
        data_mean = torch.mean(data, dim=0)
        data_var = torch.var(data, dim=0)
        self.data = (data - data_mean) / torch.sqrt(data_var)

        self.target = wineq[:, -1]  # Selects all rows and the last column

        assert len(self.data) == len(self.target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wine_feature = self.data[idx]
        wine_target = self.target[idx]
        return wine_feature, wine_target


if __name__ == "__main__":
    wine_dataset = WineDataset()

    for wine_feature, wine_target in wine_dataset:
        print("{0}: {1}".format(wine_feature.shape, wine_target.shape))

    wine_dataloader = DataLoader(
        dataset=wine_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    print()

    for wine_features, wine_targets in wine_dataloader:
        print("{0}: {1}".format(wine_features.shape, wine_targets.shape))

