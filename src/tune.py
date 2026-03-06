"""
Breast Cancer Detection - Hyperparameter Tuning
GridSearchCV on Random Forest and Logistic Regression.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib, os

os.makedirs("outputs", exist_ok=True)

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Random Forest Grid Search ────────────────────────────────
print("🔍 Tuning Random Forest...")
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth":    [None, 5, 10],
    "min_samples_split": [2, 5],
}
rf_gs = GridSearchCV(RandomForestClassifier(random_state=42),
                     rf_params, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0)
rf_gs.fit(X_train_sc, y_train)
print(f"  Best params : {rf_gs.best_params_}")
print(f"  Best CV AUC : {rf_gs.best_score_:.4f}")

# ── Logistic Regression Grid Search ─────────────────────────
print("\n🔍 Tuning Logistic Regression...")
lr_params = {
    "C":       [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver":  ["liblinear"],
}
lr_gs = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                     lr_params, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0)
lr_gs.fit(X_train_sc, y_train)
print(f"  Best params : {lr_gs.best_params_}")
print(f"  Best CV AUC : {lr_gs.best_score_:.4f}")

# ── Evaluate on Test Set ─────────────────────────────────────
print("\n── Test-set Results ──")
for name, gs in [("Random Forest", rf_gs), ("Logistic Regression", lr_gs)]:
    y_pred  = gs.best_estimator_.predict(X_test_sc)
    y_proba = gs.best_estimator_.predict_proba(X_test_sc)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n{name}  (Test AUC = {auc:.4f})")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# Save best tuned model
best_gs = rf_gs if rf_gs.best_score_ >= lr_gs.best_score_ else lr_gs
joblib.dump(best_gs.best_estimator_, "outputs/best_model_tuned.joblib")
joblib.dump(scaler, "outputs/scaler.joblib")
print("💾 Tuned model saved to outputs/best_model_tuned.joblib")
