from __future__ import annotations

from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from common import MODELS_DIR, evaluate_model, load_split, save_comparison, seed_everything


def main() -> None:
    seed_everything()
    X_train, _, X_test, y_train, _, y_test = load_split()
    baseline = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    baseline.fit(X_train, y_train)
    dump(baseline, MODELS_DIR / "gradient_boosting_baseline.joblib")
    evaluate_model("gradient_boosting_baseline", baseline, X_test, y_test, baseline.predict_proba(X_test)[:, 1], {"algorithm": "Gradient Boosting", "stage": "baseline", "optimization": "None", "parameters": baseline.get_params()})
    search = GridSearchCV(GradientBoostingClassifier(random_state=42), {"n_estimators": [100, 200], "learning_rate": [0.03, 0.1], "max_depth": [2, 3], "subsample": [0.8, 1.0]}, scoring="f1", cv=3, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    optimized = search.best_estimator_
    dump(optimized, MODELS_DIR / "gradient_boosting_optimized.joblib")
    evaluate_model("gradient_boosting_optimized", optimized, X_test, y_test, optimized.predict_proba(X_test)[:, 1], {"algorithm": "Gradient Boosting", "stage": "optimized", "optimization": "3-fold GridSearchCV with subsampling and staged boosting", "best_cv_f1": search.best_score_, "parameters": search.best_params_})
    staged_scores = []
    for probabilities in optimized.staged_predict_proba(X_test):
        from sklearn.metrics import f1_score
        staged_scores.append(f1_score(y_test, (probabilities[:, 1] >= 0.5).astype(int), zero_division=0))
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(staged_scores) + 1), staged_scores, color="#f97316")
    plt.xlabel("Boosting stage")
    plt.ylabel("Test F1")
    plt.title("Gradient Boosting optimized: F1 by boosting stage")
    plt.tight_layout()
    plt.savefig(__import__("common").FIGURES_DIR / "gradient_boosting_stage_f1.png", dpi=180)
    plt.close()
    save_comparison()


if __name__ == "__main__":
    main()