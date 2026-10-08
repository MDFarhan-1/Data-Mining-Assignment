# Online Shoppers Purchasing Intention

This project uses UCI dataset 468, **Online Shoppers Purchasing Intention**, to predict whether a browsing session ends in a purchase. The source is the UCI Machine Learning Repository through `ucimlrepo`.

## Run

Use Python 3.11 or newer. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./run_all_nohup.sh
tail -f outputs/logs/run_all.log
```

`run_all.py` downloads the raw dataset, saves UCI metadata and variable descriptions, creates a stratified 70/15/15 train/validation/test split, fits preprocessing only on training data, and trains eight distinct algorithms. Each algorithm has a normal stage and an optimized stage, producing sixteen test-set evaluations:

1. Logistic Regression: baseline and 5-fold `GridSearchCV` optimized
2. Random Forest: baseline and 3-fold `GridSearchCV` optimized
3. Gaussian Naive Bayes: baseline and 5-fold `GridSearchCV` optimized
4. Gradient Boosting: baseline and 3-fold `GridSearchCV` optimized
5. Support Vector Machine: baseline and 3-fold `GridSearchCV` optimized
6. K-Nearest Neighbors: baseline and 5-fold `GridSearchCV` optimized
7. MLP: baseline and optimized with deeper architecture, regularization, and early stopping
8. LSTM: baseline and optimized with more units, dropout, and early stopping

The positive class is `Revenue=True`. The optimized classical models choose parameters using training folds only; the test set is used once for final comparison.

## Presentation artifacts

- `data/`: raw CSV, UCI metadata, variable descriptions, frozen split, and fitted preprocessor.
- `outputs/figures/`: class balance, correlation plot, confusion matrices, and one baseline-versus-optimized comparison figure for every algorithm.
- `outputs/figures/*_metrics.png`, `outputs/figures/*_roc.png`, and `outputs/figures/*_precision_recall.png`: separate baseline-versus-optimized images for every algorithm and plot type.
- `outputs/figures/mlp_training_loss_comparison.png`, `outputs/figures/lstm_training_history.png`, and Random Forest feature importance: model-specific optimization evidence.
- `outputs/figures/logistic_optimized_coefficients.png`, `outputs/figures/gradient_boosting_stage_f1.png`, `outputs/figures/naive_bayes_variance_profile.png`, and `outputs/figures/mlp_optimized_validation_history.png`: optimization-specific plots.
- `outputs/metrics/`: one JSON and CSV per stage with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, parameters, and optimization details.
- `outputs/predictions/`: saved test labels, probabilities, and thresholded predictions used to regenerate ROC and precision-recall curves.
- `outputs/model_comparison.csv`: comparable test-set results for all sixteen stages.
- `outputs/best_model_summary.json`: the highest-F1 test-set stage.
- `outputs/models/`: fitted model files.
- `outputs/logs/run_all.log`: complete run log.

The LSTM uses the transformed feature vector as a sequence of feature steps with one channel. This is a valid dimensional model for tabular data, but it does not represent a time series because the UCI rows are independent browsing sessions. TensorFlow is optional at runtime: if unavailable, the pipeline records `lstm_skipped.json` and continues with the other models.

The requirements use TensorFlow 2.15.1 with CUDA extras. The launcher prefers `/tmp/dmdm-gpu-venv` when present, then falls back to `.venv`. It sets `CUDA_VISIBLE_DEVICES=0` and enables GPU memory growth. The LSTM trainer records the actual device in its metric JSON files. If CUDA libraries are unavailable, TensorFlow explicitly falls back to CPU and reports that fact. The LSTM trainer uses early stopping, so the graph contains the actual epochs completed rather than the configured maximum.