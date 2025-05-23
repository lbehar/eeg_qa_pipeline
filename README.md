# EEG QA + ML Pipeline (DEAP Dataset)

This project simulates a QA-tested machine learning pipeline for EEG classification using real signal structure from the DEAP dataset. It includes feature extraction, model training, evaluation, and CI/CD automation — showcasing testing best practices and explainability.

---

### Dataset Used

- **Source**: DEAP - Dataset for Emotion Analysis using Physiological signals
- **Input**: EEG features extracted from 32 scalp channels (CSV format)
- **Labels**: Simulated binary arousal labels ("high_arousal", "low_arousal")
- **Note**: Label simulation mirrors DEAP’s 1–9 arousal ratings using a thresholded normal distribution

---

### Pipeline Overview

- `mock_eeg_signals.csv`: Raw EEG features (real signals, no labels)
- `signal_filter.py`: Stubbed pre-processing (bandpass placeholder)
- `feature_extraction.py`: Extracts delta, theta, alpha, beta, gamma band powers
- `train_classifier.py`: Simulates labels, trains RandomForest, outputs metrics
- `predict.py`: Generates predictions using saved model
- `metrics.txt`: Accuracy, precision, recall, confusion matrix
- `top_features.csv`: Feature importance scores
- `top_10_features_plot.png`: Horizontal bar chart of most predictive features
- `confusion_matrix.png`: Confusion matrix visualization

---

### Tools Used

- **Python**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest
- **CI/CD**: GitHub Actions
- **Explainability**: Feature importance analysis

---

### QA & Test Coverage

- Unit tests for filtering and feature extraction
- End-to-end test for the entire pipeline
- CI pipeline runs full test suite on every push (`.github/workflows/ci.yml`)

---

### Running the Project

```bash
pip install -r requirements.txt

# Run the full pipeline
python3 processing/signal_filter.py
python3 processing/feature_extraction.py
python3 model/train_classifier.py
python3 model/predict.py

# Run tests
pytest
```

---

### Results & Insights

- Accuracy: ~48% with structured simulated labels
- Balanced classification of arousal states
- Most important EEG features: Cz_gamma, Fp1_alpha, Cz_alpha, Cz_delta, Fp1_beta
- Visuals saved: confusion_matrix.png, top_10_features_plot.png

---

### What I Learned

- Building a modular QA-friendly ML pipeline
- Simulating biologically inspired labels
- Implementing test coverage across an ML stack
- Automating with GitHub Actions
- Explaining model behavior using feature importance

---

### Future Improvements

- Load real DEAP arousal/valence ratings
- Add true bandpass filtering (e.g., Butterworth)
- Improve feature engineering
- Explore deep learning alternatives
- Wrap in a Jupyter Notebook for interactive use

---

Data format based on DEAP EEG dataset.
