import os
from pathlib import Path


# Пути
class Config:
    # Пути
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    REPORTS_DIR = BASE_DIR / "reports"

    # Параметры сигналов (для генерации синтетических данных)
    FREQ_RANGE = (800, 1000)  # MHz
    AMP_RANGE = (-70, -30)     # dB
    SNR_RANGE = (5, 30)        # dB
    BW_OPTIONS = [5, 10, 20]   # MHz

    # Создание директорий
    for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        dir_path.mkdir(exist_ok=True)


config = Config()
