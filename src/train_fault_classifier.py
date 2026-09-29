from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder


INPUT_PATH = Path("data/processed/solar_features.csv")
MODEL_PATH = Path("models/random_forest_fault_classifier.pkl")
ENCODER_PATH = Path("models/fault_label_encoder.pkl")
REPORT_PATH = Path("results/reports/fault_classifier_report.txt")

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
    "hour",
    "minute",
    "day_of_week",
    "day_of_year",
    "month",
    "is_daylight",
    "data_quality_score",
]


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = set(FEATURE_COLUMNS + ["status"])

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
            "Invalid timestamp values found."
        )

    return data.sort_values(
        "timestamp"
    ).reset_index(drop=True)


def chronological_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    split_index = int(len(data) * 0.80)

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    train_labels = set(train_data["status"])
    test_labels = set(test_data["status"])

    unseen_test_labels = sorted(
        test_labels - train_labels
    )

    if unseen_test_labels:
        print(
            "Warning: test labels not present in training data:",
            unseen_test_labels,
        )

        test_data = test_data[
            test_data["status"].isin(train_labels)
        ].copy()

    if train_data.empty:
        raise ValueError(
            "Training data is empty."
        )

    if test_data.empty:
        raise ValueError(
            "No evaluable test rows remain."
        )

    return (
        train_data,
        test_data,
        unseen_test_labels,
    )


def train_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[
    RandomForestClassifier,
    LabelEncoder,
    list[int],
    list[int],
]:
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(
        train_data["status"]
    )

    y_test = encoder.transform(
        test_data["status"]
    )

    x_train = train_data[
        FEATURE_COLUMNS
    ].replace(
        [float("inf"), float("-inf")],
        0,
    ).fillna(0)

    x_test = test_data[
        FEATURE_COLUMNS
    ].replace(
        [float("inf"), float("-inf")],
        0,
    ).fillna(0)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    return (
        model,
        encoder,
        y_test,
        y_pred,
    )


def create_report(
    data: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    unseen_test_labels: list[str],
    encoder: LabelEncoder,
    y_test: list[int],
    y_pred: list[int],
) -> str:
    labels_in_test = sorted(
        set(y_test) | set(y_pred)
    )

    target_names = encoder.inverse_transform(
        labels_in_test
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=labels_in_test,
        target_names=target_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=labels_in_test,
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report_lines = [
        "SolarSense AI - Fault Classifier Report",
        "=" * 48,
        f"Total rows: {len(data)}",
        f"Training rows: {len(train_data)}",
        f"Testing rows used: {len(test_data)}",
        "Split method: chronological 80/20",
        "",
        "Unseen labels in original test period:",
        str(unseen_test_labels),
        "",
        "Metrics",
        "-" * 25,
        f"Accuracy: {accuracy:.4f}",
        f"Balanced accuracy: {balanced_accuracy:.4f}",
        f"Weighted precision: {precision:.4f}",
        f"Weighted recall: {recall:.4f}",
        f"Weighted F1-score: {f1:.4f}",
        "",
        "Classification report",
        "-" * 25,
        report,
        "",
        "Confusion matrix labels",
        "-" * 25,
        str(list(target_names)),
        "",
        "Confusion matrix",
        "-" * 25,
        str(matrix),
        "",
        "Feature columns",
        "-" * 25,
        *FEATURE_COLUMNS,
        "",
        "Important limitations",
        "-" * 25,
        (
            "The dataset is synthetic. "
            "Metrics do not demonstrate field performance."
        ),
        (
            "Unseen labels in the test period were excluded "
            "from metric calculation because the model had no "
            "training examples for those classes."
        ),
        (
            "The classifier estimates a likely label. "
            "It does not prove physical failure."
        ),
    ]

    return "\n".join(report_lines)


def main() -> None:
    data = load_data()

    (
        train_data,
        test_data,
        unseen_test_labels,
    ) = chronological_split(data)

    (
        model,
        encoder,
        y_test,
        y_pred,
    ) = train_model(
        train_data,
        test_data,
    )

    report_text = create_report(
        data=data,
        train_data=train_data,
        test_data=test_data,
        unseen_test_labels=unseen_test_labels,
        encoder=encoder,
        y_test=y_test,
        y_pred=y_pred,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        encoder,
        ENCODER_PATH,
    )

    REPORT_PATH.write_text(
        report_text,
        encoding="utf-8",
    )

    print(report_text)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Encoder saved to: {ENCODER_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()