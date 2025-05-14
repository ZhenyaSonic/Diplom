import numpy as np
from rtlsdr import RtlSdr

from predict import SignalAnalyzer
from utils.visualization import plot_signal_quality


def capture_signal():
    """Захват сигнала с SDR устройства"""
    sdr = RtlSdr()
    sdr.sample_rate = 2.4e6
    sdr.center_freq = 915e6
    sdr.gain = 'auto'

    try:
        print("Захват сигнала...")
        samples = sdr.read_samples(1024*1024)
        power = np.mean(np.abs(samples)**2)
        snr = 10*np.log10(power/np.var(samples))
        return [sdr.center_freq/1e6, 10*np.log10(power), snr, sdr.sample_rate/1e6]
    finally:
        sdr.close()


def main():
    analyzer = SignalAnalyzer()

    while True:
        try:
            signal = capture_signal()
            report = analyzer.analyze(signal)
            print("\n" + "="*50)
            print(f"Режим реального времени - {report['timestamp']}")
            print("\n".join(report["recommendations"]))
            plot_signal_quality(signal)
        except KeyboardInterrupt:
            print("\nМониторинг остановлен")
            break


if __name__ == "__main__":
    main()
