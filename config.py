import os
from pathlib import Path


class Config:
    # Пути
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports"

    # Параметры для генерации данных
    FREQ_RANGE = (800, 1000)  # MHz
    AMP_RANGE = (-70, -30)     # dB
    SNR_RANGE = (5, 30)        # dB
    BW_OPTIONS = [5, 10, 20]   # MHz

    # Параметры для тестовых сигналов (примеры разных случаев)
    TEST_SIGNALS = [
        {
            'type': 'good',
            'freq': 915.0,
            'amp': -55.0,
            'snr': 12.0,
            'bw': 10.0
        },
        {
            'type': 'interference',
            'freq': 880.0,
            'amp': -65.0,
            'snr': 8.0,
            'bw': 20.0
        },
        {
            'type': 'noisy',
            'freq': 2450.0,
            'amp': -70.0,
            'snr': 5.0,
            'bw': 40.0
        }
    ]

    # Создание директорий
    for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        dir_path.mkdir(exist_ok=True)


config = Config()
