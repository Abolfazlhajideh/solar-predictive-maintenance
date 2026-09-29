from pathlib import Path

import joblib
import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/solar_features.csv")
OUTPUT_PATH = Path("results/exports/solar_predictions.csv")

ANOMALY_MODEL_PATH = Path("models/isolation_forest.pkl")
ANOMALY_SCALER_PATH = Path("models/anomaly_scaler.pkl")
CLASSIFIER_MODEL_PATH = Path(
    "models/random_forest_fault_classifier.pkl"
)
CLASSIFIER_ENCODER_PATH = Path(
    "models/fault_label_encoder.pkl"
)
POWER_MODEL_PATH = Path(
    "models/random_forest_power_regressor.pkl"
)

ANOMALY_FEATURE_COLUMNS = [
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

CLASSIFIER_FEATURE_COLUMNS = [
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
    "hour",
    "minute",
    "day_of_week",
    "day_of_year",
    "month",
    "is_daylight",
    "data_quality_score",
]

POWER_FEATURE_COLUMNS = [
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


def prepare_features(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    missing_columns = sorted(
        set(columns) - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    return data[columns].replace(
        [np.inf, -np.inf],
        0,
    ).fillna(0)


def calculate_risk_score(data: pd.DataFrame) -> pd.Series:
    risk = pd.Series(
        0.0,
        index=data.index,
    )

    risk += data["anomaly_flag"] * 35

    risk += (
        data["classifier_confidence"] < 0.60
    ).astype(int) * 10

    risk += (
        data["predicted_fault"] != "normal"
    ).astype(int) * 35

    risk += (
        data["panel_temperature"] > 70
    ).astype(int) * 20

    risk += (
        data["data_quality_score"] < 70
    ).astype(int) * 15

    return risk.clip(0, 100)


def assign_system_state(
    data: pd.DataFrame,
) -> pd.Series:
    states = pd.Series(
        "NORMAL",
        index=data.index,
    )

    states.loc[
        data["communication_gap_flag"] == 1
    ] = "COMMUNICATION_ERROR"

    states.loc[
        data["sensor_fault_flag"].astype(bool)
    ] = "SENSOR_FAULT"

    states.loc[
        data["panel_temperature"] > 70
    ] = "OVER_TEMPERATURE"

    states.loc[
        data["predicted_fault"] == "disconnected_panel"
    ] = "PANEL_DISCONNECTED"

    states.loc[
        data["predicted_fault"] == "shading"
    ] = "SHADING_SUSPECTED"

    states.loc[
        data["predicted_fault"] == "soiling"
    ] = "SOILING_SUSPECTED"

    states.loc[
        data["anomaly_flag"] == 1
    ] = "WARNING"

    states.loc[
        data["risk_score"] >= 70
    ] = "DANGER"

    return states


def main() -> None:
    required_paths = [
        INPUT_PATH,
        ANOMALY_MODEL_PATH,
        ANOMALY_SCALER_PATH,
        CLASSIFIER_MODEL_PATH,
        CLASSIFIER_ENCODER_PATH,
        POWER_MODEL_PATH,
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            f"Missing required files: {missing_paths}"
        )

    data = pd.read_csv(INPUT_PATH)

    anomaly_model = joblib.load(
        ANOMALY_MODEL_PATH
    )
    anomaly_scaler = joblib.load(
        ANOMALY_SCALER_PATH
    )
    classifier = joblib.load(
        CLASSIFIER_MODEL_PATH
    )
    encoder = joblib.load(
        CLASSIFIER_ENCODER_PATH
    )
    power_model = joblib.load(
        POWER_MODEL_PATH
    )

    anomaly_x = prepare_features(
        data,
        ANOMALY_FEATURE_COLUMNS,
    )
    anomaly_x_scaled = anomaly_scaler.transform(
        anomaly_x
    )

    anomaly_prediction = anomaly_model.predict(
        anomaly_x_scaled
    )
    anomaly_score = anomaly_model.decision_function(
        anomaly_x_scaled
    )

    data["anomaly_prediction"] = anomaly_prediction
    data["anomaly_flag"] = (
        anomaly_prediction == -1
    ).astype(int)
    data["anomaly_score"] = anomaly_score

    classifier_x = prepare_features(
        data,
        CLASSIFIER_FEATURE_COLUMNS,
    )

    predicted_class_ids = classifier.predict(
        classifier_x
    )

    predicted_fault = encoder.inverse_transform(
        predicted_class_ids
    )

    data["predicted_fault"] = predicted_fault

    classifier_probabilities = classifier.predict_proba(
        classifier_x
    )

    data["classifier_confidence"] = (
        classifier_probabilities.max(axis=1)
    )

    power_x = prepare_features(
        data,
        POWER_FEATURE_COLUMNS,
    )

    data["predicted_power"] = power_model.predict(
        power_x
    )

    data["prediction_residual"] = (
        data["power_dc"] - data["predicted_power"]
    )

    data["predicted_power_drop_percentage"] = np.where(
        data["predicted_power"] > 0,
        100
        * (
            data["predicted_power"]
            - data["power_dc"]
        )
        / data["predicted_power"],
        0,
    )

    data["risk_score"] = calculate_risk_score(data)
    data["system_state"] = assign_system_state(data)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Rows predicted: {len(data)}")
    print(f"Output file: {OUTPUT_PATH}")
    print("System-state distribution:")
    print(
        data["system_state"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print("Predicted-fault distribution:")
    print(
        data["predicted_fault"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()