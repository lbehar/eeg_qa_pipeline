import os
import pandas as pd
import subprocess

def test_full_pipeline_execution():
    # Run each stage of the pipeline as subprocesses
    subprocess.run(["python3", "data/generate_mock_data.py"], check=True)
    subprocess.run(["python3", "processing/signal_filter.py"], check=True)
    subprocess.run(["python3", "processing/feature_extraction.py"], check=True)
    subprocess.run(["python3", "model/train_classifier.py"], check=True)
    subprocess.run(["python3", "model/predict.py"], check=True)

    # Check output file exists
    assert os.path.exists("eeg_predictions.csv")

    # Check the predictions file has expected column
    df = pd.read_csv("eeg_predictions.csv")
    assert "predicted_state" in df.columns
    assert not df["predicted_state"].isnull().any()
