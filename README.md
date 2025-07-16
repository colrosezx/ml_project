Проект: Предсказание цены домов с использованием PyTorch



Описание

Задача: предсказать стоимость жилых домов на основе табличных данных (площадь, год постройки, расположение и пр.) из соревнования Kaggle House Prices – Advanced Regression Techniques.

В проекте реализован полный пайплайн:

Загрузка и предобработка данных (src/data_loader.py, src/preprocess.py)

Определение и сравнение архитектур полносвязных сетей (src/model.py)

Обучение и оценка моделей (src/train.py, src/utils.py)

Визуализация процесса обучения и результатов (src/visualize.py)

Демонстрация пайплайна в Jupyter-тетрадке (notebooks/main.ipynb)

Структура проекта

ml_project/
├── data/
│   ├── train.csv           # Тренировочные данные
│   ├── test.csv            # Тестовые данные
│   └── data_description.txt# Описание признаков
├── src/
│   ├── data_loader.py      # Загрузка CSV и создание DataLoader
│   ├── preprocess.py       # Предобработка: пропуски, кодирование, масштабирование
│   ├── model.py            # Определение MLP-архитектур (MLP1, MLP2)
│   ├── train.py            # Скрипт тренировки моделей и оценки
│   ├── utils.py            # Метрики (RMSE/MAE), сохранение и загрузка моделей
│   └── visualize.py        # Построение графиков loss и предсказаний
├── notebooks/
│   └── main.ipynb          # Jupyter Notebook с демонстрацией работы
├── outputs/
│   ├── figures/            # PNG-графики (loss_curve.png, pred_vs_true.png)
│   └── models/             # Сохранённые .pt веса лучшей модели
├── .gitignore
├── requirements.txt        # Необходимые зависимости
└── README.md               # Описание проекта

Установка

Клонировать репозиторий и перейти в директорию:

git clone https://github.com/colrosezx/ml_project.git
cd ml_project

Создать виртуальное окружение и установить зависимости:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
pip install -r requirements.txt

Запуск

Через Jupyter Notebook

jupyter notebook notebooks/main.ipynb

В ноутбуке последовательно выполняются ячейки:

Задание путей и импорт модулей (sys.path для src/)

Загрузка данных и их предобработка

Обучение и сравнение моделей MLP1 и MLP2

Визуализация кривых обучения и предсказаний

Сохранение лучшей модели и генерация submission.csv

Основные параметры

Архитектуры:

MLP1: один скрытый слой

MLP2: два скрытых слоя + Dropout

Функция потерь: MSELoss

Оптимизатор: Adam

Метрики: RMSE, MAE

Batch size: 32

Epochs: 30

Результаты

Loss: outputs/figures/loss.png

Prediction Plot: outputs/figures/predictions.png

Лучшая модель: outputs/models/best_model.pt

Submission: outputs/submission.csv

*Создано Людиновсковым Вадимом и Ковалевым Кириллом*
