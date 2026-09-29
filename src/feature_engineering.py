from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/sample/solar_fault_dataset.csv")
OUTPUT_PATH = Path("data/processed/solar_features.csv")

ROLLING_WINDOW = 6
SAMPLE_INTERVAL_HOURS = 5 / 60


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = pd.read_csv(path)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    return data.sort_values("timestamp").reset_index(drop=True)


def add_physical_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["power_dc"] = (
        data["panel_voltage"] * data["panel_current"]
    )

    data["energy_Wh"] = (
        data["power_dc"] * SAMPLE_INTERVAL_HOURS
    )

    data["cumulative_energy_Wh"] = (
        data.groupby(data["timestamp"].dt.date)["energy_Wh"]
        .cumsum()
    )

    solar_input_power = (
        data["irradiance"] * data["panel_area_m2"]
    )

    data["panel_efficiency"] = np.divide(
        data["power_dc"],
        solar_input_power,
        out=np.zeros(len(data)),
        where=solar_input_power.to_numpy() > 0,
    )

    data["performance_ratio"] = np.divide(
        data["power_dc"],
        data["expected_power"],
        out=np.zeros(len(data)),
        where=data["expected_power"].to_numpy() > 0,
    )

    data["power_residual"] = (
        data["power_dc"] - data["expected_power"]
    )

    data["power_drop_percentage"] = np.where(
        data["expected_power"] > 0,
        100
        * (
            data["expected_power"] - data["power_dc"]
        )
        / data["expected_power"],
        0,
    )

    data["voltage_current_ratio"] = np.divide(
        data["panel_voltage"],
        data["panel_current"],
        out=np.zeros(len(data)),
        where=data["panel_current"].to_numpy() > 0,
    )

    data["panel_ambient_temperature_difference"] = (
        data["panel_temperature"]
        - data["ambient_temperature"]
    )

    return data


def add_change_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["irradiance_change"] = (
        data["irradiance"].diff().fillna(0)
    )

    data["power_change"] = (
        data["power_dc"].diff().fillna(0)
    )

    data["power_change_rate"] = (
        data["power_change"] / SAMPLE_INTERVAL_HOURS
    )

    data["temperature_change"] = (
        data["panel_temperature"].diff().fillna(0)
    )

    return data


def add_rolling_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["rolling_power_mean"] = (
        data["power_dc"]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .mean()
    )

    data["rolling_power_median"] = (
        data["power_dc"]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .median()
    )

    data["rolling_power_std"] = (
        data["power_dc"]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .std()
        .fillna(0)
    )

    data["rolling_power_min"] = (
        data["power_dc"]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .min()
    )

    data["rolling_power_max"] = (
        data["power_dc"]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .max()
    )

    data["rolling_temperature_mean"] = (
        data["panel_temperature"]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .mean()
    )

    data["rolling_irradiance_mean"] = (
        data["irradiance"]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .mean()
    )

    data["voltage_rolling_mean"] = (
        data["panel_voltage"]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .mean()
    )

    data["current_rolling_mean"] = (
        data["panel_current"]
        .rolling(ROLLING_WINDOW, min_periods=1)
        .mean()
    )

    return data


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    timestamp = data["timestamp"]

    data["hour"] = timestamp.dt.hour
    data["minute"] = timestamp.dt.minute
    data["day_of_week"] = timestamp.dt.dayofweek
    data["day_of_year"] = timestamp.dt.dayofyear
    data["month"] = timestamp.dt.month

    data["is_daylight"] = (
        data["irradiance"] > 20
    ).astype(int)

    return data


def add_quality_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["missing_value_flag"] = (
        data.isna().any(axis=1).astype(int)
    )

    data["outlier_flag"] = (
        (data["panel_voltage"] > 30)
        | (data["panel_current"] > 20)
        | (data["panel_temperature"] > 90)
        | (data["irradiance"] > 1200)
    ).astype(int)

    data["sensor_stuck_flag"] = (
        data["panel_voltage"]
        .rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW)
        .std()
        .fillna(1)
        .eq(0)
        .astype(int)
    )

    data["communication_gap_flag"] = (
        data["timestamp"].diff()
        > pd.Timedelta(minutes=5)
    ).fillna(False).astype(int)

    data["data_quality_score"] = 100

    data.loc[
        data["missing_value_flag"] == 1,
        "data_quality_score",
    ] -= 40

    data.loc[
        data["outlier_flag"] == 1,
        "data_quality_score",
    ] -= 20

    data.loc[
        data["sensor_stuck_flag"] == 1,
        "data_quality_score",
    ] -= 20

    data.loc[
        data["communication_gap_flag"] == 1,
        "data_quality_score",
    ] -= 20

    data["data_quality_score"] = data[
        "data_quality_score"
    ].clip(0, 100)

    return data


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    data = add_physical_features(data)
    data = add_change_features(data)
    data = add_rolling_features(data)
    data = add_time_features(data)
    data = add_quality_features(data)

    return data


def main() -> None:
    data = load_data(INPUT_PATH)
    features = build_features(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print(f"Input rows: {len(data)}")
    print(f"Output rows: {len(features)}")
    print(f"Output columns: {len(features.columns)}")
    print(f"Output file: {OUTPUT_PATH}")
    print(
        "Missing values after feature engineering:",
        int(features.isna().sum().sum()),
    )


if __name__ == "__main__":
    main()