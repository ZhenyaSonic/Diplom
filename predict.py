from datetime import datetime

import joblib
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


def main():
    analyzer = SignalAnalyzer()

    while True:
        print("\n" + "="*50)
        print("Меню анализа сигналов LTE")
        print("="*50)
        print("1 - Анализ одного сигнала")
        print("2 - Пакетный анализ (5 случайных сигналов)")
        print("3 - Выход")

        choice = input("Ваш выбор (1/2/3): ")

        if choice == '1':
            analyze_single_signal(analyzer)
        elif choice == '2':
            batch_analysis(analyzer)
        elif choice == '3':
            print("Завершение работы...")
            break
        else:
            print("Неверный выбор, попробуйте снова")


if __name__ == "__main__":
    main()
