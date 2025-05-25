import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import Config
from utils.visualization import (plot_confusion_matrix,
                                 plot_interference_classes)


def load_keysight_data(filepath):
    """Загрузка и обработка данных из файла Keysight"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Находим начало данных
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == 'BEGIN':
            data_start = i + 1
            break

    # Извлекаем данные
    data = []
    for line in lines[data_start:]:
        if line.strip() == 'END':
            break
        parts = line.strip().split(',')
        if len(parts) >= 2:
            freq_hz = float(parts[0])
            power_dbm = float(parts[1])
            data.append([freq_hz, power_dbm])

    # Создаем DataFrame
    df = pd.DataFrame(data, columns=['frequency', 'peak_power'])

    # Преобразуем частоту в MHz
    df['frequency'] = df['frequency'] / 1e6

    # Генерация меток классов (в реальном проекте они должны быть в данных)
    # Здесь просто для примера - в реальности нужно разметить данные
    conditions = [
        (df['peak_power'] > -20),
        (df['peak_power'] < -40),
        (df['peak_power'].between(-40, -20))
    ]
    df['interference_type'] = np.select(
        conditions,
        Config.INTERFERENCE_TYPES,
        default='Смешанные'
    )

    processed_file = Config.DATA_DIR / "processed_data.csv"
    df.to_csv(processed_file, index=False)
    print(f"Обработанные данные сохранены в {processed_file}")
    return df


def load_or_generate_data():
    """Загрузка данных из CSV файла Keysight или генерация синтетических"""
    try:
        data_file = Config.DATA_DIR / "FILE_3.CSV.csv"
        if data_file.exists():
            print(f"Загрузка данных из {data_file}")
            return load_keysight_data(data_file)
        else:
            # Генерация синтетических данных, если файл не найден
            print("Файл не найден, генерируем синтетические данные")
            np.random.seed(42)
            data = pd.DataFrame({
                'frequency': np.random.uniform(*Config.FREQ_RANGE, 1000),
                'peak_power': np.random.uniform(*Config.PEAK_POWER_RANGE, 1000),
            })
            conditions = [
                (data['peak_power'] > -20),
                (data['peak_power'] < -40),
                (data['peak_power'].between(-40, -20))
            ]
            data['interference_type'] = np.select(
                conditions, 
                Config.INTERFERENCE_TYPES, 
                default='Смешанные'
            )
            return data
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        raise


def main():
    # Загрузка данных
    data = load_or_generate_data()

    # Проверка данных
    print("\nПервые 5 строк данных:")
    print(data.head())
    print("\nСтатистика по данным:")
    print(data.describe())
    print("\nРаспределение классов:")
    print(data['interference_type'].value_counts())

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
        random_state=42,
        stratify=y
    )

    # Обучение модели
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Оценка
    y_pred = model.predict(X_test)
    print("\nОтчет классификации:")
    print(classification_report(y_test, y_pred))

    # Визуализация
    plot_confusion_matrix(y_test, y_pred)
    plot_interference_classes(X_test, y_test, model)

    # Сохранение модели
    joblib.dump(model, Config.MODELS_DIR / "rf_interference_model.pkl")
    joblib.dump(scaler, Config.MODELS_DIR / "scaler.pkl")
    print("\nМодель и scaler сохранены в папку models")


if __name__ == "__main__":
    main()
