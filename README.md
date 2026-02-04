# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using various classification algorithms with optimized decision thresholds.

## Table of Contents

- [Overview](#overview)
- [Objective](#objective)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Results](#model-results)
  - [Baseline Model: Logistic Regression](#baseline-model-logistic-regression)
  - [Optimized Model: Logistic Regression with Threshold Selection](#optimized-model-logistic-regression-with-threshold-selection)
  - [Advanced Model: Histogram-Based Gradient Boosting (HGB)](#advanced-model-histogram-based-gradient-boosting-hgb)
- [Threshold Selection Strategy](#threshold-selection-strategy)
- [Visualizations](#visualizations)
- [Repository Structure](#repository-structure)
- [Reproducing the Results](#reproducing-the-results)

## Overview

This project implements a fraud detection system for credit card transactions using machine learning. The system is designed to maximize the detection of fraudulent transactions (recall) while maintaining an extremely low false positive rate to minimize inconvenience to legitimate customers.

## Objective

The primary objective is to **maximize recall** (identify as many fraudulent cases as possible) while keeping the **false positive rate (FPR) at or below 0.1%** (1 in 1,000 transactions).

This constraint is critical in fraud detection because:
- Missing a fraudulent transaction can result in significant financial loss
- Falsely flagging legitimate transactions creates customer friction and operational costs
- A 0.1% FPR represents an acceptable balance between detection and customer experience

## Dataset

- **Problem Type**: Binary classification for transaction-level fraud detection
- **Labels**:
  - `Fraud` (positive class) – fraudulent transaction
  - `Non-Fraud` (negative class) – legitimate transaction
- **Data Split**: Training, validation, and evaluation sets
- **Class Imbalance**: Highly imbalanced dataset typical of fraud detection scenarios

## Methodology

1. **Exploratory Data Analysis (EDA)**: Understanding data distribution and class imbalance
2. **Feature Engineering**: Preprocessing transaction attributes and creating aggregated behavior signals
3. **Model Training**: Training multiple classification models
4. **Threshold Optimization**: Custom threshold selection to meet FPR constraints
5. **Hyperparameter Tuning**: Optimizing model parameters for better performance
6. **Evaluation**: Comprehensive evaluation using precision, recall, F1-score, and confusion matrices

## Model Results

### Baseline Model: Logistic Regression

A simple logistic regression classifier serves as the baseline model.

**Performance Metrics** (at default threshold):
- **Precision**: 0.0587
- **Recall**: 0.9061
- **F1-Score**: 0.1102

**Confusion Matrix**:

| Actual \ Predicted | Non-Fraud | Fraud |
|--------------------|-----------|-------|
| **Non-Fraud**      | 221,722   | 5,729 |
| **Fraud**          | 37        | 357   |

- **True Negatives (TN)**: 221,722
- **False Positives (FP)**: 5,729 (FPR: 2.52%)
- **False Negatives (FN)**: 37
- **True Positives (TP)**: 357

*Note: The baseline model has high recall but does not meet the FPR ≤ 0.1% constraint.*

### Optimized Model: Logistic Regression with Threshold Selection

The same logistic regression model with an optimized decision threshold to meet FPR constraints.

**Threshold Selection**:
- **Chosen Threshold**: 0.9899
- **Strategy**: Max recall with FPR ≤ 0.0010

**Performance Metrics** (at optimized threshold):
- **Precision**: 0.5942
- **Recall**: 0.8325
- **F1-Score**: 0.6934
- **FPR**: 0.0010 (0.1%)

**Confusion Matrix**:

| Actual \ Predicted | Non-Fraud | Fraud |
|--------------------|-----------|-------|
| **Non-Fraud**      | 227,240   | 211   |
| **Fraud**          | 71        | 323   |

- **True Negatives (TN)**: 227,240
- **False Positives (FP)**: 211 (FPR: 0.09%)
- **False Negatives (FN)**: 71
- **True Positives (TP)**: 323

*The optimized threshold significantly reduces false positives while maintaining strong recall (83.25%).*

### Advanced Model: Histogram-Based Gradient Boosting (HGB)

A more sophisticated ensemble model using histogram-based gradient boosting with tuned hyperparameters.

**Hyperparameters** (tuned, see `models/hgb_params.json`):
- `l2_regularization`: 0.9612
- `learning_rate`: 0.1705
- `max_bins`: 181
- `max_depth`: 7
- `max_iter`: 379
- `max_leaf_nodes`: 40
- `min_samples_leaf`: 29

**Note**: Performance metrics and visualizations for HGB models are available in the `reports/` directory.

## Threshold Selection Strategy

The project implements a custom threshold selection algorithm (`src/choose_thresholds.py`) that:

1. **Computes the ROC curve** for the model's predicted probabilities
2. **Filters candidate thresholds** where FPR ≤ specified maximum (e.g., 0.001)
3. **Selects the threshold** that maximizes recall (TPR) among valid candidates
4. **Falls back** to the minimum achievable FPR if the constraint cannot be met

**Function**: `choose_threshold_max_recall_at_fpr(y_true, proba, max_fpr=0.001)`

This approach ensures that the model operates within acceptable business constraints while maximizing fraud detection capability.

## Visualizations

The following visualizations are available in the `reports/` directory:

- `cm_baseline.png` – Confusion matrix for baseline model
- `roc_auc_curve_baseline.png` – ROC curve for baseline model
- `pr_curve_baseline.png` – Precision-Recall curve for baseline model
- `pr_curve_chosen_threshold_hgb.png` – PR curve for HGB model with threshold selection
- `pr_curve_chosen_threshold_hgb_tuned.png` – PR curve for tuned HGB model
- `recall_vs_threshold_chosen_threshold_hgb.png` – Recall vs threshold for HGB model
- `recall_vs_threshold_chosen_threshold_hgb_tuned.png` – Recall vs threshold for tuned HGB model

## Repository Structure

```
.
├── data/
│   └── splits/          # Train/validation/test data splits
├── models/
│   └── hgb_params.json  # Tuned hyperparameters for HGB model
├── notebooks/
│   ├── 01_eda_split.ipynb              # Exploratory data analysis
│   ├── 02_baseline_model.ipynb         # Baseline model training
│   ├── 03_threshold.ipynb              # Threshold selection experiments
│   └── 04_hgb.ipynb                    # HGB model training and tuning
├── reports/             # Visualization outputs
├── src/
│   └── choose_thresholds.py            # Threshold selection utilities
└── README.md
```

## Reproducing the Results

To reproduce the results in this project:

1. **Data Preparation**:
   - Prepare a labeled fraud-detection dataset with binary labels (`Fraud` / `Non-Fraud`)
   - Run the EDA notebook: `notebooks/01_eda_split.ipynb`

2. **Baseline Model**:
   - Train the baseline logistic regression model: `notebooks/02_baseline_model.ipynb`
   - This will generate initial metrics and visualizations

3. **Threshold Optimization**:
   - Run the threshold selection experiments: `notebooks/03_threshold.ipynb`
   - Apply the `choose_threshold_max_recall_at_fpr` function to optimize decision boundaries

4. **Advanced Modeling**:
   - Train and tune the HGB model: `notebooks/04_hgb.ipynb`
   - Use the hyperparameters in `models/hgb_params.json` for the tuned model

5. **Evaluation**:
   - Evaluate on the held-out test set with the optimized threshold (FPR ≤ 0.1%)
   - Generate confusion matrices, ROC curves, and PR curves

All notebooks contain detailed implementation steps and can be run sequentially to reproduce the complete analysis.

---

*For questions or issues, please open an issue in the repository.*