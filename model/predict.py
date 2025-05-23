import pandas as pd
import joblib

def predict_eeg_state(feature_file="eeg_features.csv", model_file="eeg_classifier.pkl", output_file="eeg_predictions.csv"):
    """
    Load a trained model and make predictions on EEG features.
    """
    # Load features
    features = pd.read_csv(feature_file)

    # Load model
    clf = joblib.load(model_file)

    # Predict labels
    predictions = clf.predict(features)

    # Add predictions to DataFrame
    results = features.copy()
    results["predicted_state"] = predictions

    # Save to CSV
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to '{output_file}'.")

if __name__ == "__main__":
    predict_eeg_state()
