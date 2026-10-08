from __future__ import annotations

from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from common import MODELS_DIR, evaluate_model, load_split, save_comparison, seed_everything


def main() -> None:
    seed_everything()
    X_train, _, X_test, y_train, _, y_test = load_split()
    baseline = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=42)
    baseline.fit(X_train, y_train)
    dump(baseline, MODELS_DIR / "svm_baseline.joblib")
    evaluate_model("svm_baseline", baseline, X_test, y_test, baseline.predict_proba(X_test)[:, 1], {"algorithm": "Support Vector Machine", "stage": "baseline", "optimization": "None", "parameters": baseline.get_params()})
    search = GridSearchCV(CalibratedClassifierCV(SVC(class_weight="balanced", probability=False, random_state=42), cv=3, method="sigmoid"), {"estimator__C": [0.5, 1.0, 2.0], "estimator__gamma": ["scale", 0.01, 0.05], "estimator__kernel": ["rbf"]}, scoring="f1", cv=3, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    optimized = search.best_estimator_
    dump(optimized, MODELS_DIR / "svm_optimized.joblib")
    evaluate_model("svm_optimized", optimized, X_test, y_test, optimized.predict_proba(X_test)[:, 1], {"algorithm": "Support Vector Machine", "stage": "optimized", "optimization": "3-fold GridSearchCV maximizing validation F1", "best_cv_f1": search.best_score_, "parameters": search.best_params_})
    save_comparison()


if __name__ == "__main__":
    main()