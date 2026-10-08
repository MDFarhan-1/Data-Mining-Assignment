from __future__ import annotations

from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd

from common import MODELS_DIR, evaluate_model, load_split, save_comparison, seed_everything


def main() -> None:
    seed_everything()
    X_train, _, X_test, y_train, _, y_test = load_split()
    baseline = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1)
    baseline.fit(X_train, y_train)
    dump(baseline, MODELS_DIR / "random_forest_baseline.joblib")
    evaluate_model("random_forest_baseline", baseline, X_test, y_test, baseline.predict_proba(X_test)[:, 1], {"algorithm": "Random Forest", "stage": "baseline", "optimization": "None", "parameters": baseline.get_params()})
    search = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1), {"n_estimators": [200], "max_depth": [12, 18, None], "min_samples_leaf": [1, 2]}, scoring="f1", cv=3, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    optimized = search.best_estimator_
    dump(optimized, MODELS_DIR / "random_forest_optimized.joblib")
    evaluate_model("random_forest_optimized", optimized, X_test, y_test, optimized.predict_proba(X_test)[:, 1], {"algorithm": "Random Forest", "stage": "optimized", "optimization": "3-fold GridSearchCV maximizing validation F1", "best_cv_f1": search.best_score_, "parameters": search.best_params_})
    import common
    importance = pd.Series(optimized.feature_importances_).sort_values(ascending=False).head(15).sort_values()
    importance.plot(kind="barh", figsize=(8, 5), title="Random Forest optimized: top transformed features", color="#16a34a")
    plt.tight_layout()
    plt.savefig(common.FIGURES_DIR / "random_forest_optimized_feature_importance.png", dpi=180)
    plt.close()
    save_comparison()


if __name__ == "__main__":
    main()