"""Model evaluation: every number here is computed from real predictions."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """Return accuracy, precision, recall, F1, ROC-AUC, confusion matrix and ROC curve.

    Phishing (1) is the "positive" class, so:
      precision = of the URLs flagged as phishing, how many really were
      recall    = of all real phishing URLs, how many we caught
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:  # only one class present in y_test
        auc = None

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    keep = np.unique(np.linspace(0, len(fpr) - 1, num=min(len(fpr), 120)).astype(int))

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
    }


def select_best_model(results: dict) -> str:
    """Pick the model with the best F1; break ties with recall, then accuracy."""
    return max(
        results,
        key=lambda name: (results[name]["f1"], results[name]["recall"], results[name]["accuracy"]),
    )
