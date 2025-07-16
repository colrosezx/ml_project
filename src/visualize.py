"""
visualize.py

Визуализация графиков обучения и предсказаний.
"""
import matplotlib.pyplot as plt
import os


def plot_loss(train_losses, val_losses, save_path: str):
    """Строит график потерь для train и val"""
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
# Пример данных
train_losses = [1, 0.8, 0.6, 0.5]
val_losses = [1.1, 0.9, 0.7, 0.6]

plot_loss(train_losses, val_losses, 'outputs/figures/loss.png')
print("Plot saved.")

def plot_predictions(y_true, y_pred, save_path: str):
    """Сравнение реальных и предсказанных цен домов"""
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
