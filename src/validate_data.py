from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("data/sample/solar_fault_dataset.csv")
REPORT_PATH = Path("results/reports/data_validation_report.txt")

REQUIRED_COLUMNS = {
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
}

VALID_LABELS = {
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
}


def validate_columns(data: pd.DataFrame) -> list[str]:
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    unexpected_columns = sorted(set(data.columns) - REQUIRED_COLUMNS)

    messages = []

    if missing_columns:
        messages.append(f"Missing columns: {missing_columns}")
    else:
        messages.append("Required columns: PASS")

    if unexpected_columns:
        messages.append(f"Unexpected columns: {unexpected_columns}")
    else:
        messages.append("Unexpected columns: NONE")

    return messages


def validate_timestamps(data: pd.DataFrame) -> list[str]:
    messages = []

    timestamps = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    invalid_timestamp_count = int(timestamps.isna().sum())
    duplicate_timestamp_count = int(timestamps.duplicated().sum())
    unsorted = not timestamps.is_monotonic_increasing

    messages.append(
        f"Invalid timestamps: {invalid_timestamp_count}"
    )
    messages.append(
        f"Duplicate timestamps: {duplicate_timestamp_count}"
    )
    messages.append(
        f"Timestamp sorted: {not unsorted}"
    )

    return messages


def validate_missing_values(data: pd.DataFrame) -> list[str]:
    missing_counts = data.isna().sum()
    total_missing = int(missing_counts.sum())

    messages = [
        f"Total missing values: {total_missing}",
    ]

    nonzero_missing = missing_counts[missing_counts > 0]

    if not nonzero_missing.empty:
        messages.append(
            f"Columns with missing values: {nonzero_missing.to_dict()}"
        )
    else:
        messages.append("Missing-value check: PASS")

    return messages


def validate_nonnegative_values(data: pd.DataFrame) -> list[str]:
    numeric_columns = [
        "panel_voltage",
        "panel_current",
        "power_dc",
        "irradiance",
        "expected_power",
        "panel_area_m2",
    ]

    messages = []

    for column in numeric_columns:
        negative_count = int((data[column] < 0).sum())
        messages.append(
            f"{column} negative values: {negative_count}"
        )

    return messages


def validate_power_relationship(data: pd.DataFrame) -> list[str]:
    calculated_power = (
        data["panel_voltage"] * data["panel_current"]
    )

    absolute_error = np.abs(
        data["power_dc"] - calculated_power
    )

    tolerance = 0.01
    invalid_count = int((absolute_error > tolerance).sum())
    maximum_error = float(absolute_error.max())

    return [
        f"Power equation invalid rows: {invalid_count}",
        f"Maximum absolute power error: {maximum_error:.8f} W",
        f"Power relationship tolerance: {tolerance} W",
    ]


def validate_labels(data: pd.DataFrame) -> list[str]:
    observed_statuses = set(data["status"].dropna().unique())
    invalid_statuses = sorted(observed_statuses - VALID_LABELS)

    expected_fault_type = data["status"].where(
        data["status"] != "normal",
        "none",
    )

    fault_type_mismatch_count = int(
        (data["fault_type"] != expected_fault_type).sum()
    )

    return [
        f"Invalid status labels: {invalid_statuses}",
        f"Status/fault_type mismatches: {fault_type_mismatch_count}",
        f"Number of status classes: {data['status'].nunique()}",
    ]


def validate_quality_flags(data: pd.DataFrame) -> list[str]:
    valid_quality_values = {"good", "suspect"}
    observed_quality_values = set(
        data["data_quality"].dropna().unique()
    )
    invalid_quality_values = sorted(
        observed_quality_values - valid_quality_values
    )

    sensor_flag_values = set(
        data["sensor_fault_flag"].dropna().unique()
    )
    invalid_sensor_flags = sorted(
        sensor_flag_values - {True, False}
    )

    return [
        f"Invalid data_quality values: {invalid_quality_values}",
        f"Invalid sensor_fault_flag values: {invalid_sensor_flags}",
    ]


def build_report(data: pd.DataFrame) -> list[str]:
    report_lines = [
        "SolarSense AI - Data Validation Report",
        "=" * 45,
        f"Rows: {len(data)}",
        f"Columns: {len(data.columns)}",
        "",
        "Column validation",
        "-" * 25,
    ]

    report_lines.extend(validate_columns(data))

    report_lines.extend(
        [
            "",
            "Timestamp validation",
            "-" * 25,
        ]
    )
    report_lines.extend(validate_timestamps(data))

    report_lines.extend(
        [
            "",
            "Missing-value validation",
            "-" * 25,
        ]
    )
    report_lines.extend(validate_missing_values(data))

    report_lines.extend(
        [
            "",
            "Numeric validation",
            "-" * 25,
        ]
    )
    report_lines.extend(validate_nonnegative_values(data))

    report_lines.extend(
        [
            "",
            "Power relationship validation",
            "-" * 25,
        ]
    )
    report_lines.extend(validate_power_relationship(data))

    report_lines.extend(
        [
            "",
            "Label validation",
            "-" * 25,
        ]
    )
    report_lines.extend(validate_labels(data))

    report_lines.extend(
        [
            "",
            "Quality-flag validation",
            "-" * 25,
        ]
    )
    report_lines.extend(validate_quality_flags(data))

    report_lines.extend(
        [
            "",
            "Class distribution",
            "-" * 25,
        ]
    )

    report_lines.extend(
        f"{label}: {count}"
        for label, count in data["status"].value_counts().sort_index().items()
    )

    return report_lines


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    report_lines = build_report(data)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()