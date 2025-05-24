import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import Config
from utils.visualization import (plot_confusion_matrix,
                                 plot_interference_classes)


def load_or_generate_data():
    """Загрузка или генерация данных"""
    try:
        # Пытаемся загрузить реальные данные
        real_data = Config.DATA_DIR / "rf_spectrum_data.csv"
        if real_data.exists() and real_data.stat().st_size > 0:
            return pd.read_csv(real_data)

        # Генерация синтетических данных
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame({
            'frequency': np.random.uniform(*Config.FREQ_RANGE, n_samples),
            'peak_power': np.random.uniform(*Config.PEAK_POWER_RANGE, n_samples),
        })

        # Генерация меток классов
        conditions = [
            (data['peak_power'] > -20) & (data['frequency'].between(700, 1000)),
            (data['peak_power'] < -40) & (data['frequency'].between(1800, 2700)),
            (data['peak_power'].between(-40, -20)) | 
                (data['frequency'].between(1000, 1800))
        ]
        data['interference_type'] = np.select(
            conditions, 
            Config.INTERFERENCE_TYPES, 
            default='Смешанные'
        )

        data.to_csv(Config.DATA_DIR / "rf_spectrum_synthetic.csv", index=False)
        return data
    except Exception as e:
        print(f"Ошибка при работе с данными: {e}")
        raise


def main():
    # Загрузка данных
    data = load_or_generate_data()

    # Подготовка данных
    X = data[['frequency', 'peak_power']]
    y = data['interference_type']

    # Нормализация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.3,
        random_state=42
    )

    # Обучение модели (меняем на классификатор с тремя классами)
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Оценка
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)

    # Визуализация классификации (новая функция)
    plot_interference_classes(X_test, y_test, model)

    # Сохранение модели
    joblib.dump(model, Config.MODELS_DIR / "rf_interference_model.pkl")
    joblib.dump(scaler, Config.MODELS_DIR / "scaler.pkl")


if __name__ == "__main__":
    main()
