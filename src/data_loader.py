"""
data_loader.py

Модуль для загрузки данных из CSV и создания датасетов PyTorch.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def load_data(train_path: str, test_path: str):
    """
    Загружает train и test CSV в DataFrame.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


class HousePricesDataset(Dataset):
    """
    Dataset для работы с табличными данными House Prices.
    Ожидает, что признаки и целевая переменная уже предобработаны.
    """
    def __init__(self, features: torch.Tensor, targets: torch.Tensor = None):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.targets is not None:
            y = self.targets[idx]
            return x, y
        return x


def create_dataloaders(features: torch.Tensor, targets: torch.Tensor,
                       batch_size: int = 32, val_split: float = 0.2):
    """
    Разбивает данные на train и val, возвращает DataLoader-ы.
    """
    dataset = HousePricesDataset(features, targets)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader