from __future__ import annotations

from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from common import MODELS_DIR, evaluate_model, load_split, save_comparison, seed_everything


def main() -> None:
    seed_everything()
    X_train, _, X_test, y_train, _, y_test = load_split()
    baseline = KNeighborsClassifier(n_neighbors=5, weights="uniform", p=2, n_jobs=-1)
    baseline.fit(X_train, y_train)
    dump(baseline, MODELS_DIR / "knn_baseline.joblib")
    evaluate_model("knn_baseline", baseline, X_test, y_test, baseline.predict_proba(X_test)[:, 1], {"algorithm": "K-Nearest Neighbors", "stage": "baseline", "optimization": "None", "parameters": baseline.get_params()})
    search = GridSearchCV(Pipeline([("pca", PCA(whiten=True, random_state=42)), ("knn", KNeighborsClassifier(n_jobs=-1))]), {"pca__n_components": [0.90, 0.98], "knn__n_neighbors": [5, 11, 21], "knn__weights": ["uniform", "distance"], "knn__p": [1, 2]}, scoring="f1", cv=5, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    optimized = search.best_estimator_
    dump(optimized, MODELS_DIR / "knn_optimized.joblib")
    evaluate_model("knn_optimized", optimized, X_test, y_test, optimized.predict_proba(X_test)[:, 1], {"algorithm": "K-Nearest Neighbors", "stage": "optimized", "optimization": "5-fold GridSearchCV maximizing validation F1", "best_cv_f1": search.best_score_, "parameters": search.best_params_})
    save_comparison()


if __name__ == "__main__":
    main()