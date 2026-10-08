from __future__ import annotations

from joblib import dump
from sklearn.neural_network import MLPClassifier

from common import MODELS_DIR, evaluate_model, load_split, save_comparison, seed_everything


def main() -> None:
    seed_everything()
    X_train, _, X_test, y_train, _, y_test = load_split()
    baseline = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=0.0001, early_stopping=True, validation_fraction=0.15, max_iter=150, random_state=42)
    baseline.fit(X_train, y_train)
    dump(baseline, MODELS_DIR / "mlp_baseline.joblib")
    evaluate_model("mlp_baseline", baseline, X_test, y_test, baseline.predict_proba(X_test)[:, 1], {"algorithm": "Multilayer Perceptron", "stage": "baseline", "optimization": "None", "parameters": baseline.get_params()})
    optimized = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=0.001, learning_rate_init=0.001, early_stopping=True, validation_fraction=0.15, max_iter=250, random_state=42)
    optimized.fit(X_train, y_train)
    dump(optimized, MODELS_DIR / "mlp_optimized.joblib")
    evaluate_model("mlp_optimized", optimized, X_test, y_test, optimized.predict_proba(X_test)[:, 1], {"algorithm": "Multilayer Perceptron", "stage": "optimized", "optimization": "Deeper network, L2 regularization, early stopping", "parameters": optimized.get_params()})
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(baseline.loss_curve_, label="MLP baseline")
    plt.plot(optimized.loss_curve_, label="MLP optimized")
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.title("MLP training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(__import__("common").FIGURES_DIR / "mlp_training_loss_comparison.png", dpi=180)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(optimized.validation_scores_, label="validation accuracy", color="#2563eb")
    plt.plot(optimized.loss_curve_, label="training loss", color="#dc2626")
    plt.xlabel("Iteration")
    plt.ylabel("Score / loss")
    plt.title("MLP optimized: early stopping validation history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(__import__("common").FIGURES_DIR / "mlp_optimized_validation_history.png", dpi=180)
    plt.close()
    save_comparison()


if __name__ == "__main__":
    main()