from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score, recall_score, roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
LOG_DIR = OUTPUT_DIR / "logs"


def ensure_dirs() -> None:
    for directory in (DATA_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_split() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(DATA_DIR / "prepared_data.npz")
    return tuple(data[key] for key in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"))


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")


def evaluate_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray, probabilities: np.ndarray, details: Dict | None = None) -> Dict:
    ensure_dirs()
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "model": name,
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, np.column_stack([1 - probabilities, probabilities]))),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    if details:
        metrics.update(details)
    save_json(METRICS_DIR / f"{name}.json", metrics)
    np.savez(PREDICTIONS_DIR / f"{name}.npz", y_true=y_test, probabilities=probabilities, predictions=predictions)
    pd.DataFrame([{key: value for key, value in metrics.items() if key != "confusion_matrix"}]).to_csv(METRICS_DIR / f"{name}.csv", index=False)
    plt.figure(figsize=(5, 4))
    sns.heatmap(np.array(metrics["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name}: confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}_confusion_matrix.png", dpi=160)
    plt.close()
    return metrics


def save_comparison() -> None:
    rows = []
    for path in sorted(METRICS_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        if "model" in payload and payload.get("status") != "skipped":
            rows.append({key: value for key, value in payload.items() if key != "confusion_matrix"})
    if rows:
        comparison = pd.DataFrame(rows).sort_values("f1", ascending=False)
        comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
        for algorithm, group in comparison.groupby("algorithm"):
            stages = {row["stage"]: row for row in group.to_dict("records")}
            if not {"baseline", "optimized"}.issubset(stages):
                continue
            safe_name = algorithm.lower().replace(" ", "_")
            baseline = stages["baseline"]
            optimized = stages["optimized"]
            metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            baseline_scores = [baseline[name] for name in metric_names]
            optimized_scores = [optimized[name] for name in metric_names]
            positions = np.arange(len(metric_names))
            width = 0.36
            figure, axis = plt.subplots(figsize=(9, 6))
            axis.bar(positions - width / 2, baseline_scores, width, label="Baseline", color="#94a3b8")
            axis.bar(positions + width / 2, optimized_scores, width, label="Optimized", color="#2563eb")
            axis.set_xticks(positions, [name.upper() for name in metric_names], rotation=35)
            axis.set_ylim(0, 1)
            axis.set_ylabel("Test-set score")
            axis.set_title(f"{algorithm}: baseline versus optimized metrics")
            axis.legend()
            figure.tight_layout()
            figure.savefig(FIGURES_DIR / f"{safe_name}_metrics.png", dpi=180)
            plt.close(figure)

            figure, axis = plt.subplots(figsize=(8, 6))
            for stage, color in (("baseline", "#64748b"), ("optimized", "#2563eb")):
                prediction_path = PREDICTIONS_DIR / f"{stages[stage]['model']}.npz"
                prediction_data = np.load(prediction_path)
                false_positive_rate, true_positive_rate, _ = roc_curve(prediction_data["y_true"], prediction_data["probabilities"])
                axis.plot(false_positive_rate, true_positive_rate, color=color, label=f"{stage.title()} AUC={stages[stage]['roc_auc']:.3f}")
            axis.plot([0, 1], [0, 1], "k--", linewidth=1)
            axis.set_xlabel("False positive rate")
            axis.set_ylabel("True positive rate")
            axis.set_title(f"{algorithm}: baseline versus optimized ROC")
            axis.legend()
            figure.tight_layout()
            figure.savefig(FIGURES_DIR / f"{safe_name}_roc.png", dpi=180)
            plt.close(figure)

            figure, axis = plt.subplots(figsize=(8, 6))
            for stage, color in (("baseline", "#64748b"), ("optimized", "#2563eb")):
                prediction_path = PREDICTIONS_DIR / f"{stages[stage]['model']}.npz"
                prediction_data = np.load(prediction_path)
                precision, recall, _ = precision_recall_curve(prediction_data["y_true"], prediction_data["probabilities"])
                axis.plot(recall, precision, color=color, label=f"{stage.title()} F1={stages[stage]['f1']:.3f}")
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
            axis.set_title(f"{algorithm}: baseline versus optimized precision-recall")
            axis.legend()
            figure.tight_layout()
            figure.savefig(FIGURES_DIR / f"{safe_name}_precision_recall.png", dpi=180)
            plt.close(figure)
        best = comparison.iloc[0]
        (OUTPUT_DIR / "best_model_summary.json").write_text(json.dumps(best.to_dict(), indent=2, default=float) + "\n")