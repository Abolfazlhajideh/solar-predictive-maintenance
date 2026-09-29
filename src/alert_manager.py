from collections import deque
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "results/exports/solar_predictions.csv"
)

OUTPUT_PATH = Path(
    "results/exports/alerts.csv"
)

WARNING_WINDOW = 5
WARNING_REQUIRED = 3

DANGER_WINDOW = 10
DANGER_REQUIRED = 7

WARNING_RISK_THRESHOLD = 40
DANGER_RISK_THRESHOLD = 70

ALERT_COOLDOWN_MINUTES = 30


def load_predictions() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = {
        "timestamp",
        "system_state",
        "risk_score",
        "predicted_fault",
        "power_dc",
        "panel_temperature",
        "irradiance",
        "data_quality_score",
        "classifier_confidence",
    }

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


def get_warning_condition(row: pd.Series) -> bool:
    fault = str(row["predicted_fault"])
    state = str(row["system_state"])
    risk = float(row["risk_score"])

    abnormal_faults = {
        "shading",
        "soiling",
        "overheating",
        "sensor_fault",
        "abnormal_voltage",
        "abnormal_current",
        "disconnected_panel",
    }

    abnormal_states = {
        "WARNING",
        "DANGER",
        "SHADING_SUSPECTED",
        "SOILING_SUSPECTED",
        "OVER_TEMPERATURE",
        "SENSOR_FAULT",
        "PANEL_DISCONNECTED",
        "COMMUNICATION_ERROR",
    }

    return bool(
        risk >= WARNING_RISK_THRESHOLD
        or fault in abnormal_faults
        or state in abnormal_states
    )


def get_danger_condition(row: pd.Series) -> bool:
    state = str(row["system_state"])
    risk = float(row["risk_score"])

    critical_states = {
        "PANEL_DISCONNECTED",
        "COMMUNICATION_ERROR",
    }

    return bool(
        risk >= DANGER_RISK_THRESHOLD
        or state in critical_states
    )


def count_true(history: deque) -> int:
    return sum(
        bool(value)
        for value in history
    )


def has_persistent_warning(
    warning_history: deque,
) -> bool:
    return bool(
        len(warning_history) >= WARNING_WINDOW
        and count_true(warning_history)
        >= WARNING_REQUIRED
    )


def has_persistent_danger(
    danger_history: deque,
) -> bool:
    return bool(
        len(danger_history) >= DANGER_WINDOW
        and count_true(danger_history)
        >= DANGER_REQUIRED
    )


def calculate_alert_severity(
    warning_history: deque,
    danger_history: deque,
    current_warning: bool,
    current_danger: bool,
) -> str | None:
    persistent_danger = has_persistent_danger(
        danger_history
    )

    persistent_warning = has_persistent_warning(
        warning_history
    )

    if persistent_danger and current_danger:
        return "DANGER"

    if persistent_warning and current_warning:
        return "WARNING"

    return None


def recommend_action(
    severity: str,
    fault: str,
    state: str,
) -> str:
    if fault == "sensor_fault":
        return (
            "Check sensor wiring, calibration, valid range, "
            "and communication quality."
        )

    if fault == "abnormal_voltage":
        return (
            "Inspect panel voltage, connectors, string wiring, "
            "and inverter input."
        )

    if fault == "abnormal_current":
        return (
            "Inspect string current, shading, connectors, "
            "fuses, and current-sensor validity."
        )

    if fault == "overheating":
        return (
            "Check panel temperature, ventilation, mounting, "
            "hot spots, and temperature-sensor validity."
        )

    if fault == "shading":
        return (
            "Inspect nearby obstacles and compare production "
            "with irradiance."
        )

    if fault == "soiling":
        return (
            "Inspect panel surface and clean it using an "
            "approved safe procedure."
        )

    if fault == "disconnected_panel":
        return (
            "Inspect panel output, connectors, wiring, "
            "and protective devices safely."
        )

    if state == "COMMUNICATION_ERROR":
        return (
            "Check ESP32 power, Wi-Fi, API availability, "
            "and network connection."
        )

    if severity == "DANGER":
        return (
            "Request a qualified maintenance inspection. "
            "Do not treat this alert as a confirmed diagnosis."
        )

    if state == "NORMAL":
        return (
            "Continue monitoring and verify surrounding "
            "samples before taking action."
        )

    return (
        "Continue monitoring and inspect the system if the "
        "condition persists."
    )


def create_alerts(
    data: pd.DataFrame,
) -> pd.DataFrame:
    warning_history = deque(
        maxlen=WARNING_WINDOW
    )

    danger_history = deque(
        maxlen=DANGER_WINDOW
    )

    alerts = []

    for _, row in data.iterrows():
        current_warning = get_warning_condition(
            row
        )

        current_danger = get_danger_condition(
            row
        )

        warning_history.append(
            current_warning
        )

        danger_history.append(
            current_danger
        )

        severity = calculate_alert_severity(
            warning_history=warning_history,
            danger_history=danger_history,
            current_warning=current_warning,
            current_danger=current_danger,
        )

        if severity is None:
            continue

        current_state = str(
            row["system_state"]
        )

        current_fault = str(
            row["predicted_fault"]
        )

        current_risk_score = float(
            row["risk_score"]
        )

        alerts.append(
            {
                "timestamp": row["timestamp"],
                "severity": severity,
                "current_system_state": current_state,
                "persistent_alert_state": severity,
                "predicted_fault": current_fault,
                "risk_score": current_risk_score,
                "current_risk_score": current_risk_score,
                "warning_count_in_window": count_true(
                    warning_history
                ),
                "danger_count_in_window": count_true(
                    danger_history
                ),
                "current_warning_condition": int(
                    current_warning
                ),
                "current_danger_condition": int(
                    current_danger
                ),
                "power_dc": float(row["power_dc"]),
                "panel_temperature": float(
                    row["panel_temperature"]
                ),
                "irradiance": float(
                    row["irradiance"]
                ),
                "data_quality_score": float(
                    row["data_quality_score"]
                ),
                "classifier_confidence": float(
                    row["classifier_confidence"]
                ),
                "recommended_action": recommend_action(
                    severity=severity,
                    fault=current_fault,
                    state=current_state,
                ),
            }
        )

    return pd.DataFrame(alerts)


def remove_duplicate_alerts(
    alerts: pd.DataFrame,
) -> pd.DataFrame:
    if alerts.empty:
        return alerts

    alerts = alerts.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    kept_rows = []

    last_alert_key = None
    last_alert_time = None

    cooldown = pd.Timedelta(
        minutes=ALERT_COOLDOWN_MINUTES
    )

    for index, row in alerts.iterrows():
        current_key = (
            row["severity"],
            row["current_system_state"],
            row["predicted_fault"],
        )

        current_time = row["timestamp"]

        same_key = (
            current_key == last_alert_key
        )

        inside_cooldown = (
            last_alert_time is not None
            and current_time - last_alert_time
            < cooldown
        )

        if same_key and inside_cooldown:
            continue

        kept_rows.append(index)

        last_alert_key = current_key
        last_alert_time = current_time

    return alerts.loc[
        kept_rows
    ].reset_index(drop=True)


def validate_alerts(
    alerts: pd.DataFrame,
) -> None:
    if alerts.empty:
        print(
            "Validation: no alerts generated."
        )
        return

    invalid_danger_current = alerts[
        (
            alerts["severity"] == "DANGER"
        )
        & (
            alerts["current_danger_condition"] != 1
        )
    ]

    if invalid_danger_current.empty:
        print(
            "Validation: all DANGER alerts are "
            "currently dangerous."
        )
    else:
        print(
            "Validation ERROR: DANGER alerts contain "
            "non-dangerous current rows."
        )
        print(
            invalid_danger_current[
                [
                    "timestamp",
                    "current_system_state",
                    "current_risk_score",
                    "current_danger_condition",
                ]
            ].to_string(index=False)
        )

    invalid_danger_persistence = alerts[
        (
            alerts["severity"] == "DANGER"
        )
        & (
            alerts["danger_count_in_window"]
            < DANGER_REQUIRED
        )
    ]

    if invalid_danger_persistence.empty:
        print(
            "Validation: all DANGER alerts satisfy "
            "persistence requirements."
        )
    else:
        print(
            "Validation ERROR: DANGER persistence "
            "requirement failed."
        )

    invalid_warning_current = alerts[
        (
            alerts["severity"] == "WARNING"
        )
        & (
            alerts["current_warning_condition"] != 1
        )
    ]

    if invalid_warning_current.empty:
        print(
            "Validation: all WARNING alerts are "
            "currently warning conditions."
        )
    else:
        print(
            "Validation ERROR: WARNING alerts contain "
            "normal current rows."
        )

    invalid_warning_persistence = alerts[
        (
            alerts["severity"] == "WARNING"
        )
        & (
            alerts["warning_count_in_window"]
            < WARNING_REQUIRED
        )
    ]

    if invalid_warning_persistence.empty:
        print(
            "Validation: all WARNING alerts satisfy "
            "persistence requirements."
        )
    else:
        print(
            "Validation ERROR: WARNING persistence "
            "requirement failed."
        )

    normal_state_alerts = alerts[
        alerts["current_system_state"] == "NORMAL"
    ]

    print(
        "Alerts with current NORMAL state:",
        len(normal_state_alerts),
    )


def print_summary(
    data: pd.DataFrame,
    alerts: pd.DataFrame,
) -> None:
    print(
        f"Rows analyzed: {len(data)}"
    )

    print(
        f"Alerts generated: {len(alerts)}"
    )

    print(
        f"Output file: {OUTPUT_PATH}"
    )

    if alerts.empty:
        print(
            "No persistent alerts generated."
        )
        return

    print(
        "Alert severity distribution:"
    )

    print(
        alerts["severity"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "Current-state distribution:"
    )

    print(
        alerts["current_system_state"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "Fault distribution:"
    )

    print(
        alerts["predicted_fault"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "Risk summary:"
    )

    print(
        alerts.groupby("severity")[
            "current_risk_score"
        ]
        .agg(
            [
                "count",
                "min",
                "mean",
                "max",
            ]
        )
        .round(2)
        .to_string()
    )


def main() -> None:
    data = load_predictions()

    raw_alerts = create_alerts(
        data
    )

    alerts = remove_duplicate_alerts(
        raw_alerts
    )

    validate_alerts(
        alerts
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    alerts.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_summary(
        data,
        alerts,
    )


if __name__ == "__main__":
    main()