import matplotlib.pyplot as plt
import numpy as np

from config import Config


def plot_spectrum(data):
    """Визуализация распределения частот и амплитуд."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(data['frequency'], bins=30, color='blue', alpha=0.7)
    plt.title("Распределение частот")
    plt.xlabel("Частота (МГц)")
    plt.ylabel("Количество")

    plt.subplot(1, 2, 2)
    plt.hist(data['amplitude'], bins=30, color='red', alpha=0.7)
    plt.title("Распределение амплитуд")
    plt.xlabel("Амплитуда (дБ)")
    plt.show()


def generate_signal_with_params(freq=None, amp=None, snr=None, bw=None):
    """Генерация сигнала с возможностью частичного указания параметров"""
    signal = [
        freq if freq is not None else np.random.uniform(*Config.FREQ_RANGE),
        amp if amp is not None else np.random.uniform(*Config.AMP_RANGE),
        snr if snr is not None else np.random.uniform(*Config.SNR_RANGE),
        bw if bw is not None else np.random.choice(Config.BW_OPTIONS)
    ]
    return signal


def generate_signals_batch(n=5, **params):
    """Генерация набора сигналов с возможностью фиксации параметров"""
    return [generate_signal_with_params(**params) for _ in range(n)]
