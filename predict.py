from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from utils.helpers import generate_signal_with_params, generate_signals_batch
from utils.visualization import (plot_confusion_matrix,
                                 plot_interference_classes,
                                 plot_signal_quality)


class SignalAnalyzer:
    def __init__(self):
        self.model = joblib.load(Config.MODELS_DIR / "rf_interference_model.pkl")
        self.scaler = joblib.load(Config.MODELS_DIR / "scaler.pkl")

        # Устанавливаем имена признаков если их нет
        if not hasattr(self.scaler, 'feature_names_in_'):
            self.scaler.feature_names_in_ = ['frequency', 'peak_power']

    def analyze(self, signal):
        """Анализ сигнала с двумя параметрами"""
        signal_df = pd.DataFrame(
            [signal],
            columns=self.scaler.feature_names_in_
        )

        # Преобразуем и предсказываем
        signal_scaled = self.scaler.transform(signal_df)
        prediction = self.model.predict(signal_scaled)[0]
        proba = self.model.predict_proba(signal_scaled)[0]

        return {
            "frequency": signal[0],
            "peak_power": signal[1],
            "interference_type": prediction,
            "probabilities": dict(zip(Config.INTERFERENCE_TYPES, proba)),
            "timestamp": datetime.now(),
            "recommendations": self._generate_recommendations(signal, prediction)
        }

    def _generate_recommendations(self, signal, prediction):
        """Генерация рекомендаций по типу помех"""
        freq, power = signal
        recs = []

        if prediction == "Импульсные":
            recs.append("🔴 Обнаружены импульсные помехи")
            recs.append("→ Рекомендуется использовать фильтры нижних частот")
        elif prediction == "Широкополосные":
            recs.append("🟡 Обнаружены широкополосные помехи")
            recs.append("→ Рекомендуется сузить полосу пропускания")
        else:
            recs.append("🟢 Обнаружены смешанные помехи")
            recs.append("→ Требуется комплексный анализ спектра")

        if power > -20:
            recs.append("⚡ Высокая мощность сигнала - возможны искажения")
        if freq > 2500:
            recs.append("📶 Высокочастотный диапазон - возможны потери")

        return recs


def save_report(report):
    """Сохранение отчета в CSV"""
    df = pd.DataFrame([report])
    filepath = Config.REPORTS_DIR / "signal_report.csv"

    if filepath.exists():
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)
    print(f"Отчет сохранен в {filepath}")


def manual_input():
    """Ручной ввод параметров сигнала"""
    print("\n" + "="*50)
    print("Ручной ввод параметров сигнала LTE")
    print("="*50)

    freq = float(input(f"Частота (MHz) [{Config.FREQ_RANGE[0]}-{Config.FREQ_RANGE[1]}]: "))
    power = float(input(f"Пиковая мощность (dBm) [{Config.PEAK_POWER_RANGE[0]}-{Config.PEAK_POWER_RANGE[1]}]: "))

    return [freq, power]


def generate_random_signal():
    """Генерация случайного сигнала LTE"""
    print("\n" + "="*50)
    print("Генерация случайного сигнала LTE")
    print("="*50)

    freq = np.random.uniform(*Config.FREQ_RANGE)
    power = np.random.uniform(*Config.PEAK_POWER_RANGE)

    print(f"Сгенерирован сигнал: {freq:.1f} MHz, {power:.1f} dBm")
    return [freq, power]


def analyze_single_signal(analyzer):
    """Анализ одного сигнала с выбором источника"""
    print("\nВыберите источник сигнала:")
    print("1 - Ручной ввод параметров")
    print("2 - Случайная генерация сигнала")
    choice = input("Ваш выбор (1/2): ")

    if choice == '1':
        signal = manual_input()
    elif choice == '2':
        signal = generate_random_signal()
    else:
        print("Неверный выбор, используем случайный сигнал")
        signal = generate_random_signal()

    report = analyzer.analyze(signal)
    print("\n" + "="*50)
    print(f"Результаты анализа сигнала {report['frequency']} MHz:")
    print("\n".join(report["recommendations"]))
    plot_signal_quality(signal)
    save_report(report)


def batch_analysis(analyzer, n=5):
    """Пакетный анализ нескольких случайных сигналов"""
    print(f"\nГенерация и анализ {n} случайных сигналов LTE...")
    for i in range(n):
        signal = generate_random_signal()
        report = analyzer.analyze(signal)
        print(f"\nСигнал {i+1}: {report['frequency']} MHz")
        print("\n".join(report["recommendations"]))
        save_report(report)


def load_signals_from_csv(filepath):
    """Загрузка сигналов из CSV файла в формате: frequency,peak_power"""
    try:
        df = pd.read_csv(filepath)

        # Проверка необходимых столбцов
        required_columns = ['frequency', 'peak_power']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV файл должен содержать столбцы: frequency, peak_power")

        # Преобразуем в список сигналов
        signals = df[required_columns].values.tolist()
        return signals
    except Exception as e:
        print(f"Ошибка при загрузке CSV: {e}")
        return []


def analyze_csv_file(analyzer):
    """Анализ сигналов из CSV файла"""
    print("\n" + "="*50)
    print("Анализ сигналов из CSV файла")
    print("="*50)

    filepath = input("Введите путь к CSV файлу (или оставьте пустым для data/signals.csv): ").strip()
    if not filepath:
        filepath = Config.DATA_DIR / "signals.csv"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        print(f"Файл {filepath} не найден!")
        return

    signals = load_signals_from_csv(filepath)
    if not signals:
        print("Не удалось загрузить сигналы из файла")
        return

    reports = []
    for i, signal in enumerate(signals, 1):
        print(f"\nАнализ сигнала {i}/{len(signals)}: {signal[0]} MHz, {signal[1]} dBm")
        report = analyzer.analyze(signal)
        reports.append(report)

        # Вывод результатов
        print("\n".join(report["recommendations"]))
        plot_signal_quality(signal)

    # Визуализация всех сигналов на одном графике
    plot_all_signals(signals, reports)


def plot_all_signals(signals, reports):
    """Визуализация всех сигналов с зонами помех и классификацией"""
    plt.figure(figsize=(14, 8))

    try:
        # Пытаемся загрузить обучающие данные
        train_data = pd.read_csv(Config.DATA_DIR / "train_data.csv")
        plt.scatter(train_data['frequency'], train_data['peak_power'],
                   c='gray', alpha=0.2, s=30, label='Обучающие данные')
    except Exception as e:
        print(f"Не удалось загрузить обучающие данные: {e}")

    # Цвета и стили для разных типов помех
    colors = {
        'Импульсные': 'red',
        'Широкополосные': 'blue',
        'Смешанные': 'green'
    }

    # 1. Рисуем фоновые зоны помех (примерные параметры из обучения)
    plt.axhspan(-20, 0, facecolor='red', alpha=0.1, label='Зона импульсных помех')
    plt.axhspan(-50, -40, facecolor='blue', alpha=0.1, label='Зона широкополосных помех')
    plt.axhspan(-40, -20, facecolor='green', alpha=0.1, label='Зона смешанных помех')

    # 2. Рисуем реальные сигналы
    for signal, report in zip(signals, reports):
        freq, power = signal
        int_type = report['interference_type']
        plt.scatter(freq, power, color=colors[int_type], s=100, edgecolor='black', 
                   label=int_type if int_type not in plt.gca().get_legend_handles_labels()[1] else "")

    # 3. Добавляем разделительные линии и аннотации
    plt.axhline(y=-20, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=-40, color='blue', linestyle='--', alpha=0.5)

    plt.annotate('Импульсные помехи\n(высокая мощность)',
                 xy=(0.5, -10), xycoords='axes fraction',
                 ha='center', color='red')
    plt.annotate('Широкополосные помехи\n(низкая мощность)',
                 xy=(0.5, -45), xycoords='axes fraction',
                 ha='center', color='blue')
    plt.annotate('Смешанные помехи',
                 xy=(0.5, -30), xycoords='axes fraction',
                 ha='center', color='green')

    # 4. Настройки графика
    plt.xlabel('Частота (MHz)', fontsize=12)
    plt.ylabel('Пиковая мощность (dBm)', fontsize=12)
    plt.title('Классификация помех с зонами типичных параметров', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Оптимизация легенды
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    plt.legend(unique_handles, unique_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def main():
    analyzer = SignalAnalyzer()

    while True:
        print("\n" + "="*50)
        print("Меню анализа сигналов LTE")
        print("="*50)
        print("1 - Анализ одного сигнала (ручной ввод)")
        print("2 - Анализ случайного сигнала")
        print("3 - Анализ сигналов из CSV файла")
        print("4 - Пакетный анализ (5 случайных сигналов)")
        print("5 - Интерактивный анализ")
        print("6 - Выход")

        choice = input("Ваш выбор (1-5): ")

        if choice == '1':
            analyze_single_signal(analyzer)
        elif choice == '2':
            signal = generate_random_signal()
            report = analyzer.analyze(signal)
            print("\n" + "="*50)
            print(f"Результаты анализа сигнала {report['frequency']} MHz:")
            print("\n".join(report["recommendations"]))
            plot_signal_quality(signal)
            save_report(report)
        elif choice == '3':
            analyze_csv_file(analyzer)
        elif choice == '4':
            batch_analysis(analyzer)
        elif choice == '5':
            from utils.interactive_plot import interactive_analysis
            csv_path = input("Введите путь к CSV файлу (Enter для signals.csv): ").strip() or None
            interactive_analysis(csv_path)
        elif choice == '6':
            print("Завершение работы...")
            break
        else:
            print("Неверный выбор, попробуйте снова")


if __name__ == "__main__":
    main()
