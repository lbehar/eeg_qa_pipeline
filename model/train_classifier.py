import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def simulate_arousal_labels(num_samples):
    raw = np.random.normal(loc=5.0, scale=2.0, size=num_samples)
    return ["high_arousal" if x > 5 else "low_arousal" for x in raw]

def train_eeg_classifier(features_path="eeg_features.csv"):
    df = pd.read_csv(features_path)
    df['label'] = simulate_arousal_labels(len(df))

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", acc)
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

    with open("metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(matrix) + "\n")

    # Save model
    joblib.dump(clf, "eeg_classifier.pkl")
    print("Model saved to 'eeg_classifier.pkl'. Metrics saved to 'metrics.txt'.")

    # Save feature importances
    feat_importance = pd.Series(clf.feature_importances_, index=X.columns)
    feat_importance.sort_values(ascending=False).to_csv("top_features.csv")
    print("Top feature importances saved to 'top_features.csv'.")

    # Save confusion matrix plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix plot saved to 'confusion_matrix.png'.")

if __name__ == "__main__":
    train_eeg_classifier()
