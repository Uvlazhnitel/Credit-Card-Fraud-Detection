from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def choose_threshold(oof_proba, y_train, precision, recall, thresholds, target_recall=0.85):

    # Find indices where recall >= target (exclude i=0 since it has no corresponding threshold)
    mask = (recall >= target_recall)
    cand_idx = np.where(mask)[0][1:]  # Exclude i=0

    if cand_idx.size > 0:
        # Pick the candidate with max recall among those meeting precision target
        chosen_idx = cand_idx[np.argmax(precision[cand_idx])]
        chosen_thr = thresholds[chosen_idx - 1]  # Map i -> thresholds[i-1]
        strategy = f"recall≥{target_recall:.2f} → max precision"
    else:
        # Fallback: choose threshold that maximizes F1 (ignore i=0)
        f1_curve = 2 * (precision * recall) / (precision + recall + 1e-12)
        valid = np.arange(1, len(precision))  # Ignore i=0
        chosen_idx = valid[np.argmax(f1_curve[valid])]
        chosen_thr = thresholds[chosen_idx - 1]
        strategy = f"max F1 (target recall {target_recall:.2f} unattainable on OOF)"

    # Recompute metrics on the chosen threshold
    y_hat = (oof_proba >= chosen_thr).astype(int)
    metrics = {
        "precision": round(precision_score(y_train, y_hat), 6),
        "recall": round(recall_score(y_train, y_hat), 6),
        "f1": round(f1_score(y_train, y_hat), 6)
    }

    # Print strategy and chosen threshold
    print("Strategy:", strategy)
    print("Chosen index:", chosen_idx)
    print("Chosen threshold:", round(chosen_thr, 3))
    print("Point on PR: precision=", round(precision[chosen_idx], 3),
          "recall=", round(recall[chosen_idx], 3))
    print("Recomputed on OOF: ",
          "precision=", metrics["precision"],
          "recall=", metrics["recall"],
          "f1=", metrics["f1"])

    return chosen_thr, strategy, metrics