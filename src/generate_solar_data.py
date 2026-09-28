from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
SAMPLE_INTERVAL_MINUTES = 5
OUTPUT_PATH = Path("data/sample/solar_sample.csv")


def generate_solar_data() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    timestamps = pd.date_range(
        start="2026-01-15 00:00:00",
        end="2026-01-15 23:55:00",
        freq=f"{SAMPLE_INTERVAL_MINUTES}min",
    )

    hour_decimal = (
        timestamps.hour
        + timestamps.minute / 60
        + timestamps.second / 3600
    )

    daylight_factor = np.sin(
        np.pi * (hour_decimal - 6) / 12
    )
    daylight_factor = np.clip(daylight_factor, 0, None)

    clear_sky_irradiance = 900 * daylight_factor
    irradiance_noise = rng.normal(0, 18, len(timestamps))
    irradiance = np.clip(clear_sky_irradiance + irradiance_noise, 0, None)

    ambient_temperature = (
        18
        + 7 * np.sin(np.pi * (hour_decimal - 7) / 12)
        + rng.normal(0, 0.5, len(timestamps))
    )

    panel_temperature = (
        ambient_temperature
        + 0.025 * irradiance
        + rng.normal(0, 0.8, len(timestamps))
    )

    panel_area_m2 = 1.0
    panel_efficiency = 0.20

    solar_input_power = irradiance * panel_area_m2
    expected_power = solar_input_power * panel_efficiency

    panel_voltage = 18 + 1.5 * daylight_factor
    panel_voltage += rng.normal(0, 0.08, len(timestamps))
    panel_voltage = np.clip(panel_voltage, 0, None)

    panel_current = np.divide(
        expected_power,
        panel_voltage,
        out=np.zeros_like(expected_power),
        where=panel_voltage > 0,
    )
    panel_current += rng.normal(0, 0.03, len(timestamps))
    panel_current = np.clip(panel_current, 0, None)

    power_dc = panel_voltage * panel_current

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "panel_voltage": panel_voltage,
            "panel_current": panel_current,
            "power_dc": power_dc,
            "panel_temperature": panel_temperature,
            "ambient_temperature": ambient_temperature,
            "humidity": np.clip(
                70 - 25 * daylight_factor + rng.normal(0, 2, len(timestamps)),
                20,
                100,
            ),
            "irradiance": irradiance,
            "expected_power": expected_power,
            "panel_area_m2": panel_area_m2,
            "panel_efficiency": panel_efficiency,
            "status": "normal",
            "fault_type": "none",
        }
    )

    return data


def main() -> None:
    data = generate_solar_data()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated rows: {len(data)}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Columns: {len(data.columns)}")
    print(f"Maximum power: {data['power_dc'].max():.2f} W")
    print(f"Maximum irradiance: {data['irradiance'].max():.2f} W/m2")


if __name__ == "__main__":
    main()