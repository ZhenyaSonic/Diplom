import os
from pathlib import Path


class Config:
    # Пути
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports"

    # Параметры сигналов LTE
    FREQ_RANGE = (700, 2700)    # MHz (LTE диапазон)
    PEAK_POWER_RANGE = (-50, 0) # dBm (пиковая мощность)

    # Классы помех
    INTERFERENCE_TYPES = ['Импульсные', 'Широкополосные', 'Смешанные']

    # Настройки генерации
    DEFAULT_BATCH_SIZE = 5
    SIGNAL_TYPES = {
        'impulse': {'peak_power': (-10, 0)},
        'wideband': {'peak_power': (-50, -30)},
        'mixed': {'peak_power': (-30, -10)}
    }

    # Создание директорий
    for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        dir_path.mkdir(exist_ok=True)


config = Config()
