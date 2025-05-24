import matplotlib.pyplot as plt
import numpy as np

from config import Config


def plot_spectrum(data):
    """Визуализация распределения частот и мощностей"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(data['frequency'], bins=30, color='blue', alpha=0.7)
    plt.title("Распределение частот LTE")
    plt.xlabel("Частота (МГц)")
    plt.ylabel("Количество")

    plt.subplot(1, 2, 2)
    plt.hist(data['peak_power'], bins=30, color='red', alpha=0.7)
    plt.title("Распределение мощностей")
    plt.xlabel("Мощность (dBm)")
    plt.show()


def generate_signal_with_params(freq=None, power=None):
    """Генерация сигнала LTE с возможностью частичного указания параметров"""
    signal = [
        freq if freq is not None else np.random.uniform(*Config.FREQ_RANGE),
        power if power is not None else np.random.uniform(*Config.PEAK_POWER_RANGE)
    ]
    return signal


def generate_signals_batch(n=5, **params):
    """Генерация набора сигналов LTE с возможностью фиксации параметров"""
    return [generate_signal_with_params(**params) for _ in range(n)]
