import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred):
    """Визуализация матрицы ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Без помех", "Помеха"]
    )
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Матрица ошибок классификации помех")
    plt.show()


def plot_signal_quality(signal):
    """Визуализация параметров сигнала"""
    labels = ['Частота (MHz)', 'Амплитуда (dB)', 'SNR (dB)', 'Полоса (MHz)']
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, signal, color=['blue', 'green', 'orange', 'red'])

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom'
        )

    plt.title('Диагностика сигнала')
    plt.grid(axis='y', linestyle='--')
    plt.show()
