import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

def multilabel_scores(y_true, y_prob, threshold=0.5):
    """
    y_true: (N,C) 0/1
    y_prob: (N,C) logits or probs
    """

    #
    if np.max(y_prob) > 1.0 or np.min(y_prob) < 0.0:
        y_prob = 1/(1+np.exp(-y_prob))

    y_pred = (y_prob >= threshold).astype(np.int32)

    scores = {}

    #
    try:
        scores["auc_roc_macro"] = roc_auc_score(y_true, y_prob, average="macro")
    except Exception:
        scores["auc_roc_macro"] = float("nan")
    try:
        scores["auc_pr_macro"] = average_precision_score(y_true, y_prob, average="macro")
    except Exception:
        scores["auc_pr_macro"] = float("nan")

    scores["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    scores["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    scores["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    scores["accuracy"] = (y_pred == y_true).mean()
    return scores
