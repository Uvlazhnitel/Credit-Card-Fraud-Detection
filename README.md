# Fraud Detection Model Results

## Objective

The primary objective of this model is to **maximize recall** (identify as many fraudulent cases as possible) while keeping the **false positive rate (FP) at or below 0.1%**.

## Baseline Model

- **Model**: LogisticRegression (baseline)
- **Precision**: 0.0587  
- **Recall**: 0.9061  
- **F1-Score**: 0.1102  

These metrics are computed on a held-out evaluation set, focusing on high recall under a strict false-positive constraint, which is typical for fraud detection tasks where missing fraud is more costly than occasionally flagging legitimate transactions.

## Confusion Matrix

The confusion matrix below summarizes the performance of the baseline model on the evaluation dataset:

| Actual \ Predicted | Non-Fraud | Fraud |
|--------------------|-----------|-------|
| **Non-Fraud**      | 221,722   | 5,729 |
| **Fraud**          | 37        | 357   |

- **True Negatives (TN)**: 221,722  
- **False Positives (FP)**: 5,729  
- **False Negatives (FN)**: 37  
- **True Positives (TP)**: 357  

## Dataset & Methodology

- **Problem type**: Binary classification for transaction-level fraud detection.  
- **Labels**:  
  - `Fraud` – fraudulent transaction  
  - `Non-Fraud` – legitimate transaction  
- **Modeling approach**:  
  - Preprocessed input features (e.g., transaction attributes, aggregated behavior signals).  
  - Trained a baseline **LogisticRegression** classifier.  
  - Evaluation performed on a separate test set, with thresholds chosen to enforce **FP ≤ 0.1%**.  

## Reproducing the Results

To reproduce results similar to those reported here:

1. Prepare a labeled fraud-detection dataset with binary labels (`Fraud` / `Non-Fraud`).  
2. Apply the same preprocessing and feature-engineering steps used during training.  
3. Train a **LogisticRegression** model on the training portion of the data.  
4. Evaluate the model on a held-out test set, adjusting the decision threshold so that the **false positive rate is ≤ 0.1%**.  
5. Compute and report **precision**, **recall**, **F1-score**, and the **confusion matrix** as shown above.  

Refer to the project's scripts and documentation for implementation details specific to this repository.

--------------------------------------------
LogReg with chosen threshold

chosen threshold 
Chosen threshold: 0.9898731402482259 using strategy: Max recall with FPR≤0.0010
Metrics at chosen threshold:
threshold: 0.9899
fpr: 0.0010
recall: 0.8325
precision: 0.5942
f1: 0.6934

Chosen threshold cm:
Confusion Matrix:
 [[227240    211]
 [    71    323]]