
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(df):
    if "is_anomaly" not in df.columns:
        raise ValueError("Missing ground-truth column: is_anomaly")

    if "predicted_anomaly" not in df.columns:
        raise ValueError("Missing predicted_anomaly column.")

    y_true = df["is_anomaly"].astype(int)
    y_pred = df["predicted_anomaly"].astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    )

    return metrics, matrix, report
