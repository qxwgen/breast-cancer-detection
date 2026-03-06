"""
Breast Cancer Detection - Inference Script
Load the trained model and make predictions on new samples.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer

def load_artifacts(model_path="outputs/best_model.joblib",
                   scaler_path="outputs/scaler.joblib"):
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict(features: np.ndarray, model, scaler):
    """
    features : 2-D array of shape (n_samples, 30)
    Returns  : dict with prediction labels and probabilities
    """
    features_sc = scaler.transform(features)
    preds  = model.predict(features_sc)
    probas = model.predict_proba(features_sc)[:, 1]

    data = load_breast_cancer()
    labels = [data.target_names[p] for p in preds]

    return {
        "predictions":   preds.tolist(),
        "labels":        labels,
        "probabilities": probas.tolist(),   # probability of BENIGN
    }


if __name__ == "__main__":
    # Demo: run predictions on 5 test samples from the dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler as _SS

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Loading saved model...")
    model, scaler = load_artifacts()

    # Take 5 random test samples
    sample_idx = np.random.choice(len(X_test), 5, replace=False)
    X_sample   = X_test.iloc[sample_idx]
    y_sample   = y_test[sample_idx]

    results = predict(X_sample.values, model, scaler)

    print("\n── Predictions on 5 random test samples ──\n")
    for i, (pred_label, prob, true_label) in enumerate(
        zip(results["labels"], results["probabilities"], y_sample)
    ):
        true_name = data.target_names[true_label]
        match = "✅" if pred_label == true_name else "❌"
        print(f"  Sample {i+1}: Predicted={pred_label:<10}  "
              f"P(benign)={prob:.3f}  Actual={true_name}  {match}")

    print("\nDone.")
