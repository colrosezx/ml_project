"""
utils.py

Вспомогательные функции: метрики, сохранение/загрузка модели.
"""
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os


def rmse(preds: torch.Tensor) -> float:
    """Корень из MSE для вектора потерь"""
    return float(torch.sqrt(torch.mean(preds)))


def mae(preds: torch.Tensor) -> float:
    """Средняя абсолютная ошибка"""
    return float(torch.mean(torch.abs(preds)))


def save_model(model: nn.Module, path: str):
    """Сохраняет state_dict модели в файл .pt"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model: nn.Module, path: str, device: torch.device = torch.device('cpu')):
    """Загружает state_dict в модель"""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
