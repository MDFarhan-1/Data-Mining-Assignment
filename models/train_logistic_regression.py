from __future__ import annotations

from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from common import MODELS_DIR, evaluate_model, load_split, save_comparison, seed_everything


def main() -> None:
    seed_everything()
    X_train, _, X_test, y_train, _, y_test = load_split()
    baseline = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0, random_state=42)
    baseline.fit(X_train, y_train)
    dump(baseline, MODELS_DIR / "logistic_baseline.joblib")
    evaluate_model("logistic_baseline", baseline, X_test, y_test, baseline.predict_proba(X_test)[:, 1], {"algorithm": "Logistic Regression", "stage": "baseline", "optimization": "None", "parameters": baseline.get_params()})
    search = GridSearchCV(LogisticRegression(max_iter=1200, class_weight="balanced", solver="saga", penalty="elasticnet", random_state=42), {"C": [0.05, 0.15, 0.5], "l1_ratio": [0.0, 0.5, 1.0]}, scoring="f1", cv=5, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    optimized = search.best_estimator_
    dump(optimized, MODELS_DIR / "logistic_optimized.joblib")
    evaluate_model("logistic_optimized", optimized, X_test, y_test, optimized.predict_proba(X_test)[:, 1], {"algorithm": "Logistic Regression", "stage": "optimized", "optimization": "5-fold elastic-net GridSearchCV over C and L1/L2 mixture", "best_cv_f1": search.best_score_, "parameters": search.best_params_})
    top_coefficients = np.argsort(np.abs(optimized.coef_[0]))[-15:]
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(top_coefficients)), optimized.coef_[0][top_coefficients], color=np.where(optimized.coef_[0][top_coefficients] >= 0, "#16a34a", "#dc2626"))
    plt.yticks(range(len(top_coefficients)), [f"feature_{index}" for index in top_coefficients])
    plt.xlabel("Coefficient")
    plt.title("Logistic Regression optimized: strongest effects")
    plt.tight_layout()
    plt.savefig(__import__("common").FIGURES_DIR / "logistic_optimized_coefficients.png", dpi=180)
    plt.close()
    save_comparison()


if __name__ == "__main__":
    main()