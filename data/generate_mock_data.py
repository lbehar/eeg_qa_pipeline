import numpy as np
import pandas as pd

def generate_eeg_signal(freq=10, noise_level=0.2, duration=100, sampling_rate=250):
    """
    Generate a simulated EEG signal.
    freq: frequency of the sine wave (Hz)
    noise_level: standard deviation of Gaussian noise
    duration: length of signal in seconds
    sampling_rate: samples per second
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, noise_level, size=t.shape)
    return signal + noise

def create_mock_eeg_dataset():
    """Create a DataFrame with 3 channels of simulated EEG data."""
    data = {
        'Fp1': generate_eeg_signal(freq=10),  # Alpha wave
        'Fp2': generate_eeg_signal(freq=20),  # Beta wave
        'Cz': generate_eeg_signal(freq=6)     # Theta wave
    }
    df = pd.DataFrame(data)
    df.to_csv("mock_eeg_signals.csv", index=False)
    print("Mock EEG data saved to 'mock_eeg_signals.csv'.")

if __name__ == "__main__":
    create_mock_eeg_dataset()
