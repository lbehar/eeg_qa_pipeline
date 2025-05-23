import numpy as np
import pandas as pd
import pytest
from processing.signal_filter import apply_bandpass_filter

def test_apply_bandpass_filter_shape():
    # Create dummy EEG data
    df = pd.DataFrame({
        'Fp1': np.random.randn(1000),
        'Fp2': np.random.randn(1000),
        'Cz': np.random.randn(1000)
    })

    filtered = apply_bandpass_filter(df)

    # Shape should match
    assert filtered.shape == df.shape

def test_apply_bandpass_filter_no_nans_or_infs():
    df = pd.DataFrame({
        'Fp1': np.random.randn(1000),
        'Fp2': np.random.randn(1000),
        'Cz': np.random.randn(1000)
    })

    filtered = apply_bandpass_filter(df)

    assert not filtered.isnull().values.any()
    assert np.isfinite(filtered.values).all()
