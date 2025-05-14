import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from config import Config
from utils.visualization import plot_signal_quality


class SignalAnalyzer:
    def __init__(self):
        self.model = joblib.load(Config.MODELS_DIR / "rf_interference_model.pkl")
        self.scaler = joblib.load(Config.MODELS_DIR / "scaler.pkl")

    def analyze(self, signal):
        """Полный анализ сигнала"""
        signal_scaled = self.scaler.transform([signal])
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
            recs.append(
                "🔴 Высокая вероятность помех ({:.1%})"
                .format(proba[1])
            )
            if snr < 10:
                recs.append(
                    "→ Увеличить мощность передатчика (текущий SNR: {:.1f} dB)"
                    .format(snr)
                )
            if bw > 15:
                recs.append(
                    "→ Уменьшить полосу пропускания (текущая: {} MHz)"
                    .format(bw)
                )
        elif proba[1] > 0.3:
            recs.append("🟡 Возможны помехи ({:.1%})".format(proba[1]))
        else:
            recs.append("🟢 Сигнал чистый ({:.1%})".format(proba[0]))

        # Общие рекомендации
        if amp < -60:
            recs.append("⚡ Проверить антенну (амплитуда {:.1f} dB)".format(amp))
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


def main():
    analyzer = SignalAnalyzer()

    # Тестовые сигналы
    test_signals = [
        [915.0, -55.0, 12.0, 10.0],
        [880.0, -65.0, 8.0, 20.0],
        [2450.0, -70.0, 5.0, 40.0]
    ]

    for signal in test_signals:
        report = analyzer.analyze(signal)
        print(f"\nАнализ сигнала {report['frequency']} MHz:")
        print("\n".join(report["recommendations"]))
        plot_signal_quality(signal)
        save_report(report)


if __name__ == "__main__":
    main()
