"""
Breast Cancer Detection - Main Training Script
Uses the Wisconsin Breast Cancer Dataset (built into sklearn)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import joblib
import os

# ─────────────────────────────────────────────
# 1. Load & Explore Data
# ─────────────────────────────────────────────
print("=" * 60)
print("  BREAST CANCER DETECTION - ML PIPELINE")
print("=" * 60)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print(f"\n📊 Dataset shape : {X.shape}")
print(f"   Features      : {X.shape[1]}")
print(f"   Samples       : {X.shape[0]}")
print(f"\n   Class distribution:")
print(f"   Malignant (0) : {(y == 0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"   Benign    (1) : {(y == 1).sum()} ({(y==1).mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# 2. Train / Test Split & Scaling
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n🔀 Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 3. Train Multiple Models
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM (RBF kernel)":    SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}
print("\n🤖 Training models...\n")

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv  = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="accuracy").mean()

    results[name] = {"accuracy": acc, "roc_auc": auc, "cv_accuracy": cv,
                     "model": model, "y_pred": y_pred, "y_proba": y_proba}

    print(f"  ✅ {name:<25}  Acc={acc:.4f}  AUC={auc:.4f}  CV={cv:.4f}")

# ─────────────────────────────────────────────
# 4. Best Model
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best      = results[best_name]
print(f"\n🏆 Best model: {best_name}  (AUC = {best['roc_auc']:.4f})")

print("\n📋 Classification Report:")
print(classification_report(y_test, best["y_pred"],
      target_names=data.target_names))

# ─────────────────────────────────────────────
# 5. Save Plots
# ─────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

# -- Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names,
            yticklabels=data.target_names, ax=ax)
ax.set_title(f"Confusion Matrix – {best_name}", fontsize=13, fontweight="bold")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# -- ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves – All Models", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("outputs/roc_curves.png", dpi=150)
plt.close()

# -- Feature importance (Random Forest)
rf = results["Random Forest"]["model"]
importance = pd.Series(rf.feature_importances_, index=data.feature_names).nlargest(15)
fig, ax = plt.subplots(figsize=(8, 6))
importance.sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()

# -- Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
names  = list(results.keys())
accs   = [results[n]["accuracy"] for n in names]
aucs   = [results[n]["roc_auc"]  for n in names]
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, accs, w, label="Accuracy", color="steelblue")
ax.bar(x + w/2, aucs, w, label="ROC-AUC",  color="coral")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
ax.set_ylim(0.85, 1.0); ax.set_ylabel("Score")
ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.close()

print("\n📈 Plots saved to outputs/")

# ─────────────────────────────────────────────
# 6. Save Best Model & Scaler
# ─────────────────────────────────────────────
joblib.dump(best["model"], "outputs/best_model.joblib")
joblib.dump(scaler,        "outputs/scaler.joblib")
print("💾 Model & scaler saved to outputs/")
print("\n✅ Training complete!")
