import pandas as pd
import matplotlib.pyplot as plt
from mpl_interactions import panhandler, zoom_factory
from config import Config
from predict import SignalAnalyzer, load_signals_from_csv


def interactive_analysis(csv_path):
    """Интерактивный анализ сигналов из CSV"""
    analyzer = SignalAnalyzer()
    signals = load_signals_from_csv(csv_path)

    if not signals:
        return

    reports = [analyzer.analyze(signal) for signal in signals]

    fig, ax = plt.subplots(figsize=(14, 8))

    # 1. Обучающие данные (фон)
    try:
        train_data = pd.read_csv(Config.DATA_DIR / "train_data.csv")
        ax.scatter(train_data['frequency'], train_data['peak_power'],
                 c='gray', alpha=0.2, s=30, label='Обучающие данные')
    except Exception as e:
        print(f"Не удалось загрузить обучающие данные: {e}")

    # 2. Зоны помех
    ax.axhspan(-20, 0, facecolor='red', alpha=0.1, label='Импульсные зона')
    ax.axhspan(-50, -40, facecolor='blue', alpha=0.1, label='Широкополосные зона')
    ax.axhspan(-40, -20, facecolor='green', alpha=0.1, label='Смешанные зона')

    # 3. Анализируемые сигналы
    colors = {'Импульсные': 'red', 'Широкополосные': 'blue', 'Смешанные': 'green'}
    for signal, report in zip(signals, reports):
        freq, power = signal
        int_type = report['interference_type']
        ax.scatter(freq, power, color=colors[int_type], s=100, edgecolor='black')

    # Настройки графика
    ax.set_xlabel('Частота (MHz)')
    ax.set_ylabel('Пиковая мощность (dBm)')
    ax.set_title('Интерактивный анализ помех')
    ax.grid(True)
    ax.legend()

    # Добавляем интерактивность
    panhandler(fig)
    zoom_factory(ax)

    plt.show()


if __name__ == "__main__":
    csv_path = input("Введите путь к CSV файлу: ")
    interactive_analysis(csv_path)
