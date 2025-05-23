import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid

def compute_band_power(freqs, psd, band):
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return trapezoid(psd[idx_band], freqs[idx_band]) if np.any(idx_band) else 0.0

def extract_features(eeg_data, sampling_rate=250, window_size=500):
    """
    Extract features in overlapping windows.
    window_size: number of samples per window (e.g., 500 = 2 seconds)
    """
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }

    num_windows = len(eeg_data) // window_size
    all_features = []

    for w in range(num_windows):
        window = eeg_data.iloc[w * window_size:(w + 1) * window_size]
        features = {}
        for channel in window.columns:
            freqs, psd = welch(window[channel], fs=sampling_rate)
            for band, rng in bands.items():
                features[f"{channel}_{band}"] = compute_band_power(freqs, psd, rng)
        all_features.append(features)

    return pd.DataFrame(all_features)

if __name__ == "__main__":
    df = pd.read_csv("filtered_eeg_signals.csv")
    feature_df = extract_features(df)
    feature_df.to_csv("eeg_features.csv", index=False)
    print("EEG features saved to 'eeg_features.csv'.")
