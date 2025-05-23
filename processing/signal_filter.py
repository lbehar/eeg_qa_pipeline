import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Create a bandpass Butterworth filter.
    lowcut: Low cutoff frequency in Hz
    highcut: High cutoff frequency in Hz
    fs: Sampling frequency in Hz
    order: Filter order
    """
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=250.0, order=5):
    """
    Apply a bandpass filter to each channel in a DataFrame.
    data: DataFrame with EEG channels as columns
    Returns a filtered DataFrame
    """
    from pandas import DataFrame
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered_data = {col: filtfilt(b, a, data[col]) for col in data.columns}
    return DataFrame(filtered_data)

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("mock_eeg_signals.csv")
    filtered_df = apply_bandpass_filter(df)
    filtered_df.to_csv("filtered_eeg_signals.csv", index=False)
    print("Filtered EEG data saved to 'filtered_eeg_signals.csv'.")
