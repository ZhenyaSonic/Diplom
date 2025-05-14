import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import Config
from utils.visualization import plot_confusion_matrix


def load_or_generate_data():
    """Загрузка или генерация данных"""
    try:
        # Пытаемся загрузить реальные данные
        real_data = Config.DATA_DIR / "rf_spectrum_data.csv"
        if real_data.exists() and real_data.stat().st_size > 0:
            return pd.read_csv(real_data)

        # Генерация синтетических данных
        np.random.seed(42)
        data = pd.DataFrame({
            'frequency': np.random.uniform(*Config.FREQ_RANGE, 1000),
            'amplitude': np.random.uniform(*Config.AMP_RANGE, 1000),
            'snr': np.random.uniform(*Config.SNR_RANGE, 1000),
            'bandwidth': np.random.choice(Config.BW_OPTIONS, 1000),
            'is_interference': np.random.randint(0, 2, 1000)
        })
        data.to_csv(Config.DATA_DIR / "rf_spectrum_synthetic.csv", index=False)
        return data
    except Exception as e:
        print(f"Ошибка при работе с данными: {e}")
        raise


def main():
    # Загрузка данных
    data = load_or_generate_data()

    # Подготовка данных
    X = data[['frequency', 'amplitude', 'snr', 'bandwidth']]
    y = data['is_interference']

    # Нормализация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.3,
        random_state=42
    )

    # Обучение модели
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight={0: 1, 1: 2},  # Больший вес для помех
        random_state=42
    )
    model.fit(X_train, y_train)

    # Оценка
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)

    # Сохранение модели
    joblib.dump(model, Config.MODELS_DIR / "rf_interference_model.pkl")
    joblib.dump(scaler, Config.MODELS_DIR / "scaler.pkl")


if __name__ == "__main__":
    main()
