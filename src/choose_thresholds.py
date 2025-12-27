import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def choose_threshold(oof_proba, y_train, precision, recall, thresholds, target_recall=0.85):
    # precision, recall: length = len(thresholds)+1  (as in precision_recall_curve)

    # Valid indices for thresholds are 1..len(precision)-1
    valid_idx = np.arange(1, len(precision))

    # candidates: recall >= target among valid indices
    cand_idx = valid_idx[recall[valid_idx] >= target_recall]

    if cand_idx.size > 0:
        # choose max precision among candidates
        chosen_idx = cand_idx[np.argmax(precision[cand_idx])]
        strategy = f"recall≥{target_recall:.2f} → max precision"
    else:
        f1_curve = 2 * (precision * recall) / (precision + recall + 1e-12)
        chosen_idx = valid_idx[np.argmax(f1_curve[valid_idx])]
        strategy = f"max F1 (target recall {target_recall:.2f} unattainable on OOF)"

    chosen_thr = thresholds[chosen_idx - 1]

    y_hat = (oof_proba >= chosen_thr).astype(int)
    metrics = {
        "precision": float(precision_score(y_train, y_hat)),
        "recall": float(recall_score(y_train, y_hat)),
        "f1": float(f1_score(y_train, y_hat)),
    }

    print("Strategy:", strategy)
    print("Chosen threshold:", round(float(chosen_thr), 3))
    print("PR point: precision=", round(float(precision[chosen_idx]), 3),
          "recall=", round(float(recall[chosen_idx]), 3))
    print("Recomputed on OOF:", metrics)

    return chosen_thr, strategy, metrics
