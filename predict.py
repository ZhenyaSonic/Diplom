import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from config import Config
from utils.visualization import plot_signal_quality


class SignalAnalyzer:
    def __init__(self):
        self.model = joblib.load(
            Config.MODELS_DIR /
            "rf_interference_model.pkl"
        )
        self.scaler = joblib.load(
            Config.MODELS_DIR /
            "scaler.pkl"
        )

    def analyze(self, signal):
        """Полный анализ сигнала"""
        # Преобразуем в DataFrame с правильными именами признаков
        signal_df = pd.DataFrame([signal], columns=[
            'frequency',
            'amplitude',
            'snr',
            'bandwidth']
        )
        signal_scaled = self.scaler.transform(signal_df)

        proba = self.model.predict_proba(signal_scaled)[0]
        prediction = self.model.predict(signal_scaled)[0]

        return {
            "frequency": signal[0],
            "amplitude": signal[1],
            "snr": signal[2],
            "bandwidth": signal[3],
            "is_interference": bool(prediction),
            "interference_prob": float(proba[1]),
            "timestamp": datetime.now(),
            "recommendations": self._generate_recommendations(signal, proba)
        }

    def _generate_recommendations(self, signal, proba):
        """Генерация рекомендаций по улучшению"""
        freq, amp, snr, bw = signal
        recs = []

        if proba[1] > 0.7:
            recs.append(f"🔴 Высокая вероятность помех ({proba[1]:.1%})")
            if snr < 10:
                recs.append(f"→ Увеличить мощность передатчика (SNR: {snr:.1f} dB)")
            if bw > 15:
                recs.append(f"→ Уменьшить полосу пропускания ({bw} MHz)")
        elif proba[1] > 0.3:
            recs.append(f"🟡 Возможны помехи ({proba[1]:.1%})")
        else:
            recs.append(f"🟢 Сигнал чистый ({proba[0]:.1%})")

        if amp < -60:
            recs.append(f"⚡ Проверить антенну (амплитуда {amp:.1f} dB)")
        if 2400 < freq < 2500:
            recs.append("📶 Wi-Fi диапазон может быть перегружен")

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
    print("Ручной ввод параметров сигнала")
    print("="*50)

    freq = float(input("Частота (MHz): "))
    amp = float(input("Амплитуда (dB): "))
    snr = float(input("SNR (dB): "))
    bw = float(input("Ширина полосы (MHz): "))

    return [freq, amp, snr, bw]


def generate_random_signal():
    """Генерация случайного сигнала"""
    print("\n" + "="*50)
    print("Генерация случайного сигнала")
    print("="*50)

    freq = np.random.uniform(*Config.FREQ_RANGE)
    amp = np.random.uniform(*Config.AMP_RANGE)
    snr = np.random.uniform(*Config.SNR_RANGE)
    bw = np.random.choice(Config.BW_OPTIONS)

    print(f"Сгенерирован сигнал: {freq:.1f} MHz, {amp:.1f} dB, SNR {snr:.1f} dB, полоса {bw} MHz")
    return [freq, amp, snr, bw]


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
    print(f"\nГенерация и анализ {n} случайных сигналов...")
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
        print("Меню анализа сигналов")
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
