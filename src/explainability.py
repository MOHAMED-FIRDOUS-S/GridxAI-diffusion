
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS


def explain_anomalies(df, top_n=10):
    required_columns = [
        "node_id",
        "timestamp",
        "anomaly_score",
        "predicted_anomaly",
        *FEATURE_COLUMNS,
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    results = df.copy()

    def explain_row(row):
        deviations = {
            feature: abs(row[f"{feature}_scaled"])
            for feature in FEATURE_COLUMNS
            if f"{feature}_scaled" in row.index
        }

        if not deviations:
            return "Scaled feature values are unavailable."

        strongest_feature = max(deviations, key=deviations.get)

        return (
            f"Strongest deviation: {strongest_feature} "
            f"(absolute standardized value "
            f"{deviations[strongest_feature]:.2f})."
        )

    results["explanation"] = results.apply(explain_row, axis=1)

    flagged = results[results["predicted_anomaly"] == 1]

    return (
        flagged.sort_values("anomaly_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
