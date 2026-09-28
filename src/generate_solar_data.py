from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
SAMPLE_INTERVAL_MINUTES = 5
OUTPUT_PATH = Path("data/sample/solar_fault_dataset.csv")

PANEL_AREA_M2 = 1.0
NOMINAL_EFFICIENCY = 0.20
NOMINAL_VOLTAGE = 18.0


FAULT_LABELS = [
    "normal",
    "shading",
    "soiling",
    "overheating",
    "sensor_fault",
    "disconnected_panel",
    "wiring_fault",
    "low_irradiance",
    "abnormal_voltage",
    "abnormal_current",
]


def create_time_index() -> pd.DatetimeIndex:
    return pd.date_range(
        start="2026-01-15 00:00:00",
        end="2026-01-17 23:55:00",
        freq=f"{SAMPLE_INTERVAL_MINUTES}min",
    )


def create_base_weather(
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    hour_decimal = (
        timestamps.hour
        + timestamps.minute / 60
        + timestamps.second / 3600
    )

    daylight_factor = np.sin(
        np.pi * (hour_decimal - 6) / 12
    )
    daylight_factor = np.clip(daylight_factor, 0, None)

    clear_sky_irradiance = 950 * daylight_factor
    cloud_factor = np.ones(len(timestamps))

    cloud_windows = [
        ("2026-01-15 10:00:00", "2026-01-15 11:30:00", 0.55),
        ("2026-01-16 13:00:00", "2026-01-16 14:30:00", 0.40),
        ("2026-01-17 09:00:00", "2026-01-17 10:00:00", 0.65),
    ]

    for start, end, factor in cloud_windows:
        mask = (timestamps >= start) & (timestamps <= end)
        cloud_factor[mask] = factor

    irradiance = clear_sky_irradiance * cloud_factor
    irradiance += rng.normal(0, 15, len(timestamps))
    irradiance = np.clip(irradiance, 0, None)

    ambient_temperature = (
        17
        + 8 * np.sin(np.pi * (hour_decimal - 7) / 12)
        + rng.normal(0, 0.6, len(timestamps))
    )

    panel_temperature = (
        ambient_temperature
        + 0.028 * irradiance
        + rng.normal(0, 0.8, len(timestamps))
    )

    humidity = np.clip(
        72 - 28 * daylight_factor + rng.normal(0, 2, len(timestamps)),
        20,
        100,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "daylight_factor": daylight_factor,
            "irradiance": irradiance,
            "ambient_temperature": ambient_temperature,
            "panel_temperature": panel_temperature,
            "humidity": humidity,
        }
    )


def assign_fault_labels(
    timestamps: pd.DatetimeIndex,
) -> pd.Series:
    labels = pd.Series("normal", index=timestamps)

    scenarios = [
        (
            "shading",
            "2026-01-15 09:00:00",
            "2026-01-15 10:30:00",
        ),
        (
            "soiling",
            "2026-01-15 13:00:00",
            "2026-01-15 16:00:00",
        ),
        (
            "overheating",
            "2026-01-16 11:00:00",
            "2026-01-16 12:30:00",
        ),
        (
            "sensor_fault",
            "2026-01-16 15:00:00",
            "2026-01-16 16:00:00",
        ),
        (
            "disconnected_panel",
            "2026-01-17 10:00:00",
            "2026-01-17 10:30:00",
        ),
        (
            "wiring_fault",
            "2026-01-17 11:30:00",
            "2026-01-17 13:00:00",
        ),
        (
            "low_irradiance",
            "2026-01-17 14:00:00",
            "2026-01-17 15:30:00",
        ),
        (
            "abnormal_voltage",
            "2026-01-15 14:00:00",
            "2026-01-15 15:00:00",
        ),
        (
            "abnormal_current",
            "2026-01-16 09:00:00",
            "2026-01-16 10:00:00",
        ),
    ]

    for label, start, end in scenarios:
        mask = (timestamps >= start) & (timestamps <= end)
        labels.loc[mask] = label

    return labels.reset_index(drop=True)


def apply_fault_effects(
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    data = data.copy()

    solar_input_power = (
        data["irradiance"]
        * PANEL_AREA_M2
        * NOMINAL_EFFICIENCY
    )

    data["expected_power"] = solar_input_power

    data["panel_voltage"] = (
        NOMINAL_VOLTAGE
        + 1.5 * data["daylight_factor"]
        + rng.normal(0, 0.08, len(data))
    )

    data["panel_current"] = np.divide(
        data["expected_power"],
        data["panel_voltage"],
        out=np.zeros(len(data)),
        where=data["panel_voltage"].to_numpy() > 0,
    )

    labels = data["status"]

    shading_mask = labels == "shading"
    data.loc[shading_mask, "expected_power"] *= 0.45
    data.loc[shading_mask, "panel_current"] *= 0.45

    soiling_mask = labels == "soiling"
    data.loc[soiling_mask, "expected_power"] *= 0.70
    data.loc[soiling_mask, "panel_current"] *= 0.70

    overheating_mask = labels == "overheating"
    data.loc[overheating_mask, "panel_temperature"] += 22
    data.loc[overheating_mask, "expected_power"] *= 0.82
    data.loc[overheating_mask, "panel_current"] *= 0.82

    sensor_fault_mask = labels == "sensor_fault"
    data.loc[sensor_fault_mask, "panel_voltage"] += 5
    data.loc[sensor_fault_mask, "panel_current"] *= 0.20

    disconnected_mask = labels == "disconnected_panel"
    data.loc[disconnected_mask, "panel_voltage"] = 0
    data.loc[disconnected_mask, "panel_current"] = 0
    data.loc[disconnected_mask, "expected_power"] = 0

    wiring_mask = labels == "wiring_fault"
    data.loc[wiring_mask, "panel_current"] *= 0.35
    data.loc[wiring_mask, "expected_power"] *= 0.35

    low_irradiance_mask = labels == "low_irradiance"
    data.loc[low_irradiance_mask, "irradiance"] *= 0.25
    data.loc[low_irradiance_mask, "expected_power"] *= 0.25
    data.loc[low_irradiance_mask, "panel_current"] *= 0.25

    abnormal_voltage_mask = labels == "abnormal_voltage"
    data.loc[abnormal_voltage_mask, "panel_voltage"] *= 1.35

    abnormal_current_mask = labels == "abnormal_current"
    data.loc[abnormal_current_mask, "panel_current"] *= 1.50

    data["panel_voltage"] += rng.normal(0, 0.05, len(data))
    data["panel_current"] += rng.normal(0, 0.02, len(data))

    data["panel_voltage"] = np.clip(
        data["panel_voltage"],
        0,
        None,
    )
    data["panel_current"] = np.clip(
        data["panel_current"],
        0,
        None,
    )

    data["power_dc"] = (
        data["panel_voltage"]
        * data["panel_current"]
    )

    data["data_quality"] = "good"
    data["sensor_fault_flag"] = False

    data.loc[sensor_fault_mask, "data_quality"] = "suspect"
    data.loc[sensor_fault_mask, "sensor_fault_flag"] = True

    dropout_mask = (
        data.index.to_series()
        .groupby(data.index // 100)
        .transform(lambda group: False)
    )
    data.loc[dropout_mask, "data_quality"] = "good"

    return data


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    timestamps = create_time_index()
    data = create_base_weather(timestamps, rng)

    data["status"] = assign_fault_labels(timestamps)
    data["fault_type"] = data["status"]

    data = apply_fault_effects(data, rng)

    data["panel_area_m2"] = PANEL_AREA_M2
    data["panel_efficiency"] = np.divide(
        data["power_dc"],
        data["irradiance"] * data["panel_area_m2"],
        out=np.zeros(len(data)),
        where=data["irradiance"].to_numpy() > 0,
    )

    data["fault_type"] = data["status"].where(
        data["status"] != "normal",
        "none",
    )

    column_order = [
        "timestamp",
        "panel_voltage",
        "panel_current",
        "power_dc",
        "panel_temperature",
        "ambient_temperature",
        "humidity",
        "irradiance",
        "expected_power",
        "panel_area_m2",
        "panel_efficiency",
        "data_quality",
        "sensor_fault_flag",
        "status",
        "fault_type",
    ]

    return data[column_order]


def main() -> None:
    data = generate_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated rows: {len(data)}")
    print(f"Output file: {OUTPUT_PATH}")
    print("Class distribution:")
    print(data["status"].value_counts().sort_index().to_string())
    print(f"Maximum power: {data['power_dc'].max():.2f} W")
    print(f"Maximum irradiance: {data['irradiance'].max():.2f} W/m2")


if __name__ == "__main__":
    main()