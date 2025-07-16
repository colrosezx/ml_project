"""
preprocess.py

Модуль для предобработки: работы с пропущенными значениями,
кодирования категорий и нормализации числовых признаков.
"""
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Обрабатывает train и test DataFrame:
    - Заполняет пропуски
    - Кодирует категориальные признаки
    - Масштабирует числовые
    Возвращает torch.Tensor для признаков и целей.
    """
    # 1) Объединяем для одинаковых преобразований
    all_df = pd.concat([train_df.drop('SalePrice', axis=1), test_df], sort=False)

    # 2) Пропуски: числовые - медиана, категориальные - отдельная категория 'Missing'
    num_cols = all_df.select_dtypes(include=['int64','float64']).columns
    cat_cols = all_df.select_dtypes(include=['object']).columns

    for c in num_cols:
        all_df[c].fillna(all_df[c].median(), inplace=True)
    for c in cat_cols:
        all_df[c].fillna('Missing', inplace=True)

    # 3) Кодирование категорий One-Hot
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_ohe = ohe.fit_transform(all_df[cat_cols])

    # 4) Масштабирование числовых
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(all_df[num_cols])

    # 5) Объединяем признаки
    features = torch.tensor(
        np.hstack([num_scaled, cat_ohe]), dtype=torch.float32)

    # 6) Делим обратно на train и test
    n_train = len(train_df)
    train_feats = features[:n_train]
    test_feats = features[n_train:]
    train_targets = torch.tensor(train_df['SalePrice'].values, dtype=torch.float32).unsqueeze(1)

    return train_feats, train_targets, test_feats
