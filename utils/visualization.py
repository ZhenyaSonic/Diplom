import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import Config


def plot_confusion_matrix(y_true, y_pred):
    """Визуализация матрицы ошибок для трех классов"""
    cm = confusion_matrix(y_true, y_pred, labels=Config.INTERFERENCE_TYPES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.INTERFERENCE_TYPES,
                yticklabels=Config.INTERFERENCE_TYPES)
    plt.title('Матрица ошибок классификации помех')
    plt.ylabel('Истинные классы')
    plt.xlabel('Предсказанные классы')
    plt.show()


def plot_signal_quality(signal):
    """Визуализация параметров сигнала LTE"""
    labels = ['Частота (MHz)', 'Пиковая мощность (dBm)']
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, signal, color=['blue', 'orange'])

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom'
        )

    plt.title('Параметры сигнала LTE')
    plt.grid(axis='y', linestyle='--')
    plt.show()


def plot_interference_classes(X, y, model):
    """Визуализация классификации помех в стиле Fuzzy KNN"""
    plt.figure(figsize=(10, 8))

    # Сетка для отображения областей решений
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Предсказание для каждой точки сетки
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([Config.INTERFERENCE_TYPES.index(z) for z in Z])
    Z = Z.reshape(xx.shape)

    # Цветовые области
    plt.contourf(xx, yy, Z, alpha=0.4, 
                levels=len(Config.INTERFERENCE_TYPES),
                colors=['red', 'green', 'blue'])

    # Настоящие точки данных
    for i, type_name in enumerate(Config.INTERFERENCE_TYPES):
        idx = np.where(np.array(y) == type_name)
        plt.scatter(X[idx, 0], X[idx, 1], c=[plt.cm.Set1(i)], 
                   label=type_name, edgecolor='black')

    plt.xlabel('Частота LTE (MHz)')
    plt.ylabel('Пиковая мощность (dBm)')
    plt.title('Классификация типов помех в сигналах LTE')
    plt.legend()
    plt.grid(True)
    plt.show()
