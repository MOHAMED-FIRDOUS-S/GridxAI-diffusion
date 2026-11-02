
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "voltage",
    "current",
    "frequency",
    "power"
]

REQUIRED_COLUMNS = [
    "timestamp",
    "node_id",
    *FEATURE_COLUMNS,
    "is_anomaly"
]


def preprocess_data(df):
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )

    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data["is_anomaly"] = pd.to_numeric(
        data["is_anomaly"], errors="coerce"
    )

    data = data.dropna(
        subset=["timestamp", "node_id", "is_anomaly"]
    )

    data = data.dropna(
        subset=FEATURE_COLUMNS, how="all"
    )

    data[FEATURE_COLUMNS] = data.groupby(
        "node_id"
    )[FEATURE_COLUMNS].transform(
        lambda group: group.fillna(group.median())
    )

    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].fillna(
        data[FEATURE_COLUMNS].median()
    )

    if data.empty:
        raise ValueError("No valid sensor readings remain.")

    scaler = StandardScaler()

    data[[f"{col}_scaled" for col in FEATURE_COLUMNS]] = (
        scaler.fit_transform(data[FEATURE_COLUMNS])
    )

    return data.reset_index(drop=True)
