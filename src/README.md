# 🩺 Breast Cancer Detection – Machine Learning Pipeline

Detect breast cancer (malignant vs. benign) using the classic
**Wisconsin Breast Cancer Dataset** (built into scikit-learn, 569 samples, 30 features).

---

## 📁 Project Structure

```
breast_cancer_detection/
├── train.py          # Full training pipeline (4 models)
├── predict.py        # Load saved model & run inference
├── eda.py            # Exploratory data analysis + plots
├── tune.py           # GridSearchCV hyperparameter tuning
├── requirements.txt  # Python dependencies
└── outputs/          # Auto-created – holds plots & saved models
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Explore the data
```bash
python eda.py
# Generates: outputs/eda_*.png
```

### 3. Train models
```bash
python train.py
# Trains: Logistic Regression, Random Forest, Gradient Boosting, SVM
# Generates: outputs/*.png, outputs/best_model.joblib, outputs/scaler.joblib
```

### 4. Tune hyperparameters (optional)
```bash
python tune.py
# Generates: outputs/best_model_tuned.joblib
```

### 5. Run inference
```bash
python predict.py
# Loads saved model and runs predictions on 5 random test samples
```

---

## 🤖 Models Compared

| Model | Typical Accuracy | Notes |
|-------|-----------------|-------|
| Logistic Regression | ~97% | Fast, interpretable baseline |
| Random Forest | ~96–98% | Best feature importance |
| Gradient Boosting | ~97–98% | Strong overall performer |
| SVM (RBF) | ~97–98% | Excellent with scaled features |

---

## 📊 Features

The dataset has **30 numeric features** computed from digitised images of
fine needle aspirate (FNA) of breast masses:

- **Mean, SE, and Worst** values of:
  radius, texture, perimeter, area, smoothness,
  compactness, concavity, concave points, symmetry, fractal dimension

**Target classes:**
- `0` = Malignant
- `1` = Benign

---

## 📈 Outputs

After running `train.py` the `outputs/` folder contains:

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Confusion matrix of best model |
| `roc_curves.png` | ROC curves for all 4 models |
| `feature_importance.png` | Top 15 features (Random Forest) |
| `model_comparison.png` | Accuracy & AUC bar chart |
| `best_model.joblib` | Serialised best model |
| `scaler.joblib` | Fitted StandardScaler |

---

## 🔬 Dataset Reference

> W.N. Street, W.H. Wolberg and O.L. Mangasarian.
> *Nuclear feature extraction for breast tumor diagnosis.*
> IS&T/SPIE 1993 International Symposium on Electronic Imaging, 1993.

Available via `sklearn.datasets.load_breast_cancer()`.
