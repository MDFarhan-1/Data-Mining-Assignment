from __future__ import annotations

from joblib import dump
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

from common import MODELS_DIR, evaluate_model, load_split, save_comparison, seed_everything


def main() -> None:
    seed_everything()
    X_train, _, X_test, y_train, _, y_test = load_split()
    baseline = GaussianNB()
    baseline.fit(X_train, y_train)
    dump(baseline, MODELS_DIR / "naive_bayes_baseline.joblib")
    evaluate_model("naive_bayes_baseline", baseline, X_test, y_test, baseline.predict_proba(X_test)[:, 1], {"algorithm": "Gaussian Naive Bayes", "stage": "baseline", "optimization": "None", "parameters": baseline.get_params()})
    search = GridSearchCV(GaussianNB(), {"var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]}, scoring="f1", cv=5, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    optimized = search.best_estimator_
    dump(optimized, MODELS_DIR / "naive_bayes_optimized.joblib")
    evaluate_model("naive_bayes_optimized", optimized, X_test, y_test, optimized.predict_proba(X_test)[:, 1], {"algorithm": "Gaussian Naive Bayes", "stage": "optimized", "optimization": "5-fold variance-smoothing GridSearchCV plus calibrated posterior analysis", "best_cv_f1": search.best_score_, "parameters": search.best_params_})
    variance = optimized.var_.mean(axis=0)
    plt.figure(figsize=(8, 5))
    plt.hist(np.log10(variance + 1e-12), bins=20, color="#7c3aed")
    plt.xlabel("log10(mean feature variance)")
    plt.ylabel("Number of transformed features")
    plt.title("Naive Bayes optimized: variance profile")
    plt.tight_layout()
    plt.savefig(__import__("common").FIGURES_DIR / "naive_bayes_variance_profile.png", dpi=180)
    plt.close()
    save_comparison()


if __name__ == "__main__":
    main()