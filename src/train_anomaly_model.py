from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/processed/solar_features.csv")
MODEL_PATH = Path("models/isolation_forest.pkl")
SCALER_PATH = Path("models/anomaly_scaler.pkl")
OUTPUT_PATH = Path("results/reports/anomaly_training_report.txt")

FEATURE_COLUMNS = [
    "panel_voltage",
    "panel_current",
    "power_dc",
    "panel_temperature",
    "ambient_temperature",
    "humidity",
    "irradiance",
    "expected_power",
    "power_residual",
    "power_drop_percentage",
    "panel_ambient_temperature_difference",
    "irradiance_change",
    "power_change",
    "power_change_rate",
    "rolling_power_mean",
    "rolling_power_std",
    "rolling_temperature_mean",
    "rolling_irradiance_mean",
    "voltage_rolling_mean",
    "current_rolling_mean",
]


def load_training_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    missing_features = sorted(
        set(FEATURE_COLUMNS) - set(data.columns)
    )

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    return data


def main() -> None:
    data = load_training_data()

    x = data[FEATURE_COLUMNS].copy()
    x = x.replace([float("inf"), float("-inf")], 0)
    x = x.fillna(0)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_scaled)

    predictions = model.predict(x_scaled)
    anomaly_scores = model.decision_function(x_scaled)

    data["anomaly_prediction"] = predictions
    data["anomaly_flag"] = (predictions == -1).astype(int)
    data["anomaly_score"] = anomaly_scores

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    report_lines = [
        "SolarSense AI - Isolation Forest Training Report",
        "=" * 52,
        f"Rows used: {len(data)}",
        f"Features used: {len(FEATURE_COLUMNS)}",
        f"Detected anomalies: {int(data['anomaly_flag'].sum())}",
        f"Anomaly rate: {data['anomaly_flag'].mean():.4f}",
        "",
        "Training note:",
        (
            "This is an unsupervised anomaly detector. "
            "Anomaly does not necessarily mean physical failure."
        ),
        "",
        "Feature columns:",
        *FEATURE_COLUMNS,
    ]

    OUTPUT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()