from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"\n=== Running {script} ===", flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True, env=environment)


def main() -> None:
    run("download_dataset.py")
    run("prepare_data.py")
    for script in (
        "models/train_logistic_regression.py",
        "models/train_random_forest.py",
        "models/train_naive_bayes.py",
        "models/train_gradient_boosting.py",
        "models/train_svm.py",
        "models/train_knn.py",
        "models/train_mlp.py",
        "models/train_lstm.py",
    ):
        run(script)
    print(f"\nComplete. Presentation artifacts are in {ROOT / 'outputs'}")


if __name__ == "__main__":
    main()