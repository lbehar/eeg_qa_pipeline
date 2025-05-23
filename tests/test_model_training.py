import os
import pandas as pd
import numpy as np
import joblib
from model.train_classifier import simulate_arousal_labels as simulate_labels


def test_simulate_labels_validity():
    labels = simulate_labels(100)
    valid_labels = {"normal", "seizure", "drowsy", "focused"}
    assert set(labels).issubset(valid_labels)

def test_model_training_and_saving(tmp_path):
    # Create fake feature data
    feature_data = pd.DataFrame({
        'Fp1_delta': np.random.rand(10),
        'Fp1_theta': np.random.rand(10),
        'Fp1_alpha': np.random.rand(10),
        'Fp1_beta': np.random.rand(10),
        'Fp1_gamma': np.random.rand(10),
        'Fp2_delta': np.random.rand(10),
        'Fp2_theta': np.random.rand(10),
        'Fp2_alpha': np.random.rand(10),
        'Fp2_beta': np.random.rand(10),
        'Fp2_gamma': np.random.rand(10),
        'Cz_delta': np.random.rand(10),
        'Cz_theta': np.random.rand(10),
        'Cz_alpha': np.random.rand(10),
        'Cz_beta': np.random.rand(10),
        'Cz_gamma': np.random.rand(10),
    })
    test_csv = tmp_path / "mock_features.csv"
    feature_data.to_csv(test_csv, index=False)

    # Train classifier
    train_eeg_classifier(features_path=str(test_csv))

    # Check if model was saved
    assert os.path.exists("eeg_classifier.pkl")
    clf = joblib.load("eeg_classifier.pkl")
    assert hasattr(clf, "predict")
