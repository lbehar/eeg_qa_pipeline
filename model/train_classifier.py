import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def simulate_labels(num_samples):
    """Simulate realistic EEG mental state labels."""
    states = ["normal", "seizure", "drowsy", "focused"]
    return np.random.choice(states, size=num_samples)

def train_eeg_classifier(features_path="eeg_features.csv"):
    """Train a multiclass classifier on EEG features."""
    df = pd.read_csv(features_path)

    # Simulate labels for now
    df['label'] = simulate_labels(len(df))

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(clf, "eeg_classifier.pkl")
    print("Model saved to 'eeg_classifier.pkl'.")

if __name__ == "__main__":
    train_eeg_classifier()
