from __future__ import annotations

import json

import numpy as np

from common import FIGURES_DIR, METRICS_DIR, MODELS_DIR, evaluate_model, load_split, save_comparison, seed_everything


def main() -> None:
    try:
        import tensorflow as tf
    except Exception as error:
        payload = {"model": "lstm", "status": "skipped", "reason": "TensorFlow is not installed; install requirements.txt to run this model."}
        payload["error"] = str(error)
        (METRICS_DIR / "lstm_skipped.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(payload["reason"])
        return
    seed_everything()
    tf.random.set_seed(42)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    device_name = gpus[0].name if gpus else "CPU"
    if not gpus:
        print("No TensorFlow GPU registered; training on CPU.")
    X_train, X_val, X_test, y_train, y_val, y_test = load_split()
    X_train, X_val, X_test = X_train[..., np.newaxis], X_val[..., np.newaxis], X_test[..., np.newaxis]

    def build(units: int, dropout: float, advanced: bool = False):
        if advanced:
            layers = [tf.keras.layers.Input(shape=X_train.shape[1:]), tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True, dropout=dropout)), tf.keras.layers.BatchNormalization(), tf.keras.layers.LSTM(units // 2, dropout=dropout), tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dropout(dropout), tf.keras.layers.Dense(1, activation="sigmoid")]
        else:
            layers = [tf.keras.layers.Input(shape=X_train.shape[1:]), tf.keras.layers.LSTM(units, dropout=dropout), tf.keras.layers.Dense(1, activation="sigmoid")]
        model = tf.keras.Sequential(layers)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])
        return model

    callbacks = lambda: [tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-5), tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True)]
    class_weights = {0: 1.0, 1: float((y_train == 0).sum() / max((y_train == 1).sum(), 1))}
    baseline = build(32, 0.0)
    baseline.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=128, verbose=0, callbacks=callbacks(), class_weight=class_weights)
    baseline.save(MODELS_DIR / "lstm_baseline.keras")
    evaluate_model("lstm_baseline", baseline, X_test, y_test, baseline.predict(X_test, verbose=0).ravel(), {"algorithm": "LSTM", "stage": "baseline", "optimization": "None", "device": device_name, "parameters": {"units": 32, "dropout": 0.0, "loss": "binary_crossentropy"}})
    optimized = build(64, 0.25, advanced=True)
    history = optimized.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=128, verbose=0, callbacks=callbacks(), class_weight=class_weights)
    optimized.save(MODELS_DIR / "lstm_optimized.keras")
    evaluate_model("lstm_optimized", optimized, X_test, y_test, optimized.predict(X_test, verbose=0).ravel(), {"algorithm": "LSTM", "stage": "optimized", "optimization": "Bidirectional stacked LSTM, batch normalization, dropout, class weighting, adaptive learning rate, and early stopping", "device": device_name, "parameters": {"units": 64, "dropout": 0.25, "loss": "binary_crossentropy", "class_weights": class_weights}})
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("LSTM optimized training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lstm_training_history.png", dpi=180)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["auc"], label="training AUC")
    plt.plot(history.history["val_auc"], label="validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("LSTM optimized: validation AUC and early stopping")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lstm_optimized_auc_history.png", dpi=180)
    plt.close()
    np.savez(FIGURES_DIR / "lstm_training_history.npz", **history.history)
    save_comparison()


if __name__ == "__main__":
    main()