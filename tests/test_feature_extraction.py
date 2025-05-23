import numpy as np
import pandas as pd
import pytest
from processing.feature_extraction import extract_features

def test_feature_extraction_output_shape():
    df = pd.DataFrame({
        'Fp1': np.random.randn(1000),
        'Fp2': np.random.randn(1000),
        'Cz': np.random.randn(1000)
    })
    features = extract_features(df)

    # Should return one row per channel, 5 features per channel
    assert features.shape[1] == 3 * 5  # 3 channels x 5 bands

def test_feature_extraction_no_nans_or_infs():
    df = pd.DataFrame({
        'Fp1': np.random.randn(1000),
        'Fp2': np.random.randn(1000),
        'Cz': np.random.randn(1000)
    })
    features = extract_features(df)
    assert not features.isnull().values.any()
    assert np.isfinite(features.values).all()

def test_feature_extraction_column_naming():
    df = pd.DataFrame({
        'Fp1': np.random.randn(1000)
    })
    features = extract_features(df)
    expected_columns = ['Fp1_delta', 'Fp1_theta', 'Fp1_alpha', 'Fp1_beta', 'Fp1_gamma']
    assert all(col in features.columns for col in expected_columns)
