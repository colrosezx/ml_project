"""
train.py

Тренировка и оценка моделей на основе PyTorch.
"""
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from src.model import MLP1, MLP2
from src.utils import save_model, load_model, rmse, mae


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 50, lr: float = 1e-3, weight_decay: float = 0.0,
                device: torch.device = torch.device('cpu')) -> tuple:
    """
    Обучает модель, сохраняет лучшую по RMSE на валидации.
    Возвращает обученную модель, список train_loss по эпохам и список val_loss по эпохам.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_rmse = float('inf')

    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, num_epochs + 1):
        # Тренировочная фаза
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Средний loss за эпоху на тренировке
        train_loss = sum(train_losses) / len(train_losses)

        # Валидационная фаза
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                val_losses.append(criterion(preds, y_batch).item())

        # Средний loss за эпоху на валидации
        val_loss = sum(val_losses) / len(val_losses)

        # RMSE на валидации для метрики
        val_rmse = rmse(torch.tensor(val_losses, device=device))

        # Сохраняем историю для графиков
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_rmse={val_rmse:.4f}")

        # Сохраняем лучшую модель по RMSE
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            save_model(model, 'outputs/models/best_model.pt')

    return model, train_loss_history, val_loss_history