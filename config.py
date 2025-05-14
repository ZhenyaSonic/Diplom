import os
from pathlib import Path


class Config:
    # Пути
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports"

    # Параметры сигналов
    FREQ_RANGE = (800, 1000)  # MHz
    AMP_RANGE = (-70, -30)    # dB
    SNR_RANGE = (5, 30)       # dB
    BW_OPTIONS = [5, 10, 20]  # MHz

    # Настройки генерации
    DEFAULT_BATCH_SIZE = 5
    SIGNAL_TYPES = {
        'good': {'snr': (15, 30), 'amp': (-50, -30)},
        'noisy': {'snr': (5, 15), 'amp': (-70, -50)},
        'interference': {'bw': 20}
    }

    # Создание директорий
    for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        dir_path.mkdir(exist_ok=True)


config = Config()
