import numpy as np
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score

def choose_threshold_max_recall_at_fpr(y_true, proba, max_fpr=0.001):
    fpr, tpr, thr = roc_curve(y_true, proba)  # tpr == recall

    ok = np.where(fpr <= max_fpr)[0]
    if ok.size == 0:
        # If it's impossible, pick the minimum FPR point (closest to constraint)
        i = np.argmin(fpr)
        strategy = f"Could not reach FPR≤{max_fpr:.4f}; picked min-FPR threshold"
    else:
        i = ok[np.argmax(tpr[ok])]
        strategy = f"Max recall with FPR≤{max_fpr:.4f}"

    chosen_thr = thr[i]
    y_hat = (proba >= chosen_thr).astype(int)

    metrics = {
        "threshold": float(chosen_thr),
        "fpr": float(fpr[i]),
        "recall": float(recall_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
    }
    return chosen_thr, strategy, metrics,
