"""
Breast Cancer Detection - Exploratory Data Analysis
Run this script to generate EDA plots before training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import os

os.makedirs("outputs", exist_ok=True)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="diagnosis")   # 0=malignant, 1=benign
df = pd.concat([X, y], axis=1)

print("Dataset Info")
print(df.describe().T[["mean", "std", "min", "max"]].round(3).to_string())

# ── 1. Class balance ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
counts = y.value_counts()
ax.bar(data.target_names, [counts[0], counts[1]], color=["tomato", "steelblue"])
ax.set_title("Class Distribution", fontweight="bold")
ax.set_ylabel("Count")
for i, v in enumerate([counts[0], counts[1]]):
    ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda_class_balance.png", dpi=150)
plt.close()

# ── 2. Correlation heatmap (mean features) ───────────────────
mean_cols = [c for c in X.columns if "mean" in c]
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df[mean_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation (mean features)", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/eda_correlation.png", dpi=150)
plt.close()

# ── 3. Distribution of top 6 mean features ───────────────────
top6 = ["mean radius", "mean texture", "mean perimeter",
        "mean area", "mean smoothness", "mean concavity"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, feat in zip(axes.flat, top6):
    for label, color in zip([0, 1], ["tomato", "steelblue"]):
        subset = df[df.diagnosis == label][feat]
        ax.hist(subset, bins=25, alpha=0.6,
                label=data.target_names[label], color=color)
    ax.set_title(feat, fontweight="bold")
    ax.legend(fontsize=8)
plt.suptitle("Feature Distributions by Class", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda_distributions.png", dpi=150)
plt.close()

# ── 4. Boxplots ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, feat in zip(axes.flat, top6):
    data_plot = [df[df.diagnosis == 0][feat].values,
                 df[df.diagnosis == 1][feat].values]
    bp = ax.boxplot(data_plot, patch_artist=True,
                    labels=data.target_names)
    bp["boxes"][0].set_facecolor("tomato")
    bp["boxes"][1].set_facecolor("steelblue")
    ax.set_title(feat, fontweight="bold")
plt.suptitle("Feature Boxplots by Class", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda_boxplots.png", dpi=150)
plt.close()

print("\n✅ EDA plots saved to outputs/")
