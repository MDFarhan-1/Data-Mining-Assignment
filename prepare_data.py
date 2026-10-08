from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import DATA_DIR, FIGURES_DIR, ensure_dirs, seed_everything

SEED = 42
TARGET = "Revenue"


def main() -> None:
    ensure_dirs()
    seed_everything(SEED)
    source = DATA_DIR / "online_shoppers_purchasing_intention.csv"
    if not source.exists():
        raise FileNotFoundError("Raw data is missing. Run download_dataset.py first.")
    frame = pd.read_csv(source)
    frame[TARGET] = frame[TARGET].astype(str).str.lower().map({"true": 1, "false": 0})
    frame = frame.dropna(subset=[TARGET])
    X = frame.drop(columns=[TARGET])
    y = frame[TARGET].astype("int64")
    categorical = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    numerical = [column for column in X.columns if column not in categorical]
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformer = ColumnTransformer([("numeric", numeric_pipe, numerical), ("categorical", categorical_pipe, categorical)])
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp_raw, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)
    X_train = transformer.fit_transform(X_train_raw).astype("float32")
    X_val = transformer.transform(X_val_raw).astype("float32")
    X_test = transformer.transform(X_test_raw).astype("float32")
    import numpy as np
    np.savez(DATA_DIR / "prepared_data.npz", X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train.to_numpy(), y_val=y_val.to_numpy(), y_test=y_test.to_numpy())
    dump(transformer, DATA_DIR / "preprocessor.joblib")
    summary = {"rows": int(len(frame)), "original_features": int(X.shape[1]), "transformed_features": int(X_train.shape[1]), "train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test)), "positive_rate": float(y.mean()), "categorical_features": categorical, "numeric_features": numerical, "seed": SEED}
    (DATA_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    frame[TARGET].value_counts().rename(index={0: "No purchase", 1: "Purchase"}).plot(kind="bar", color=["#3b82f6", "#f97316"], title="Purchase intention class balance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance.png", dpi=160)
    plt.close()
    plt.figure(figsize=(10, 8))
    sns.heatmap(frame.select_dtypes(include="number").corr(numeric_only=True), cmap="vlag", center=0)
    plt.title("Numeric feature correlations")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_correlations.png", dpi=160)
    plt.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()