from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


INPUT_PATH = Path("data/processed/solar_features.csv")
MODEL_PATH = Path("models/random_forest_power_regressor.pkl")
REPORT_PATH = Path("results/reports/power_model_report.txt")

TARGET_COLUMN = "power_dc"

FEATURE_COLUMNS = [
    "irradiance",
    "panel_temperature",
    "ambient_temperature",
    "humidity",
    "panel_voltage",
    "panel_current",
    "expected_power",
    "rolling_irradiance_mean",
    "rolling_temperature_mean",
    "rolling_power_mean",
    "hour",
    "minute",
    "day_of_week",
    "day_of_year",
    "month",
    "is_daylight",
]


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = set(
        FEATURE_COLUMNS + [TARGET_COLUMN, "timestamp"]
    )

    missing_columns = sorted(
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    if data["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps found."
        )

    return data.sort_values(
        "timestamp"
    ).reset_index(drop=True)


def calculate_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    nonzero_mask = y_true != 0

    if not np.any(nonzero_mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (
                    y_true[nonzero_mask]
                    - y_pred[nonzero_mask]
                )
                / y_true[nonzero_mask]
            )
        )
        * 100
    )


def main() -> None:
    data = load_data()

    split_index = int(len(data) * 0.80)

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    x_train = train_data[
        FEATURE_COLUMNS
    ].replace(
        [np.inf, -np.inf],
        0,
    ).fillna(0)

    x_test = test_data[
        FEATURE_COLUMNS
    ].replace(
        [np.inf, -np.inf],
        0,
    ).fillna(0)

    y_train = train_data[TARGET_COLUMN]
    y_test = test_data[TARGET_COLUMN]

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )
    r2 = r2_score(y_test, y_pred)
    mape = calculate_mape(
        y_test.to_numpy(),
        y_pred,
    )
    explained_variance = explained_variance_score(
        y_test,
        y_pred,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)

    report_lines = [
        "SolarSense AI - Power Regression Report",
        "=" * 48,
        f"Total rows: {len(data)}",
        f"Training rows: {len(train_data)}",
        f"Testing rows: {len(test_data)}",
        "Split method: chronological 80/20",
        "",
        f"MAE: {mae:.4f} W",
        f"RMSE: {rmse:.4f} W",
        f"R2: {r2:.4f}",
        f"MAPE: {mape:.4f}%",
        f"Explained variance: {explained_variance:.4f}",
        "",
        "Feature columns:",
        *FEATURE_COLUMNS,
        "",
        "Limitations:",
        (
            "The dataset is synthetic and the regression "
            "metrics do not demonstrate field performance."
        ),
        (
            "The model predicts short-term expected power. "
            "It does not predict physical component failure."
        ),
    ]

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()