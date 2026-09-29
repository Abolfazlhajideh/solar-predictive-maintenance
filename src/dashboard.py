from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT
    / "results"
    / "exports"
    / "solar_predictions.csv"
)

ALERTS_PATH = (
    ROOT
    / "results"
    / "exports"
    / "alerts.csv"
)


st.set_page_config(
    page_title="SolarPulse PV Command Center",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLORS = {
    "green": "#00D4A8",
    "blue": "#4EA1FF",
    "yellow": "#F6B73C",
    "red": "#FF5C77",
    "white": "#F2F7FC",
    "muted": "#91A4B7",
    "background": "#050B13",
    "surface": "#0C1725",
    "surface_2": "#112235",
    "border": "#213B53",
    "grid": "#22394E",
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(
                    circle at 0% 0%,
                    rgba(0, 212, 168, 0.11),
                    transparent 26%
                ),
                radial-gradient(
                    circle at 100% 0%,
                    rgba(78, 161, 255, 0.10),
                    transparent 24%
                ),
                {COLORS["background"]};
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background:
                linear-gradient(
                    180deg,
                    #07111D 0%,
                    #091A2A 55%,
                    #07111D 100%
                );
            border-right: 1px solid {COLORS["border"]};
        }}

        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.2rem;
        }}

        .block-container {{
            max-width: 1680px;
            padding-top: 0.9rem;
            padding-bottom: 3rem;
        }}

        .sidebar-title {{
            color: {COLORS["white"]};
            font-size: 1.22rem;
            font-weight: 900;
            letter-spacing: -0.04em;
        }}

        .sidebar-subtitle {{
            color: #A8BBCB;
            font-size: 0.75rem;
            line-height: 1.45;
            margin-top: 0.3rem;
        }}

        .sidebar-live {{
            color: {COLORS["green"]};
            font-size: 0.7rem;
            font-weight: 850;
            letter-spacing: 0.07em;
            margin-top: 0.55rem;
        }}

        .sidebar-section {{
            color: {COLORS["white"]};
            font-size: 0.82rem;
            font-weight: 850;
            letter-spacing: 0.035em;
            margin-top: 1.05rem;
            margin-bottom: 0.45rem;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid #1B344A;
        }}

        .sidebar-note {{
            color: #A8BBCB;
            font-size: 0.68rem;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }}

        .sidebar-file {{
            color: #A8BBCB;
            font-size: 0.67rem;
            line-height: 1.45;
            padding: 0.45rem 0;
            border-bottom: 1px solid #172D40;
            overflow-wrap: anywhere;
        }}

        .sidebar-footer {{
            color: #62798D;
            font-size: 0.65rem;
            line-height: 1.45;
            margin-top: 1.1rem;
            padding-top: 0.75rem;
            border-top: 1px solid #1B344A;
        }}

        .header {{
            padding: 1.2rem 1.35rem;
            border: 1px solid {COLORS["border"]};
            border-radius: 22px;
            background:
                linear-gradient(
                    135deg,
                    rgba(12, 23, 37, 0.98),
                    rgba(12, 34, 51, 0.95)
                );
            box-shadow: 0 18px 50px rgba(0,0,0,0.22);
        }}

        .title {{
            color: {COLORS["white"]};
            font-size: clamp(1.55rem, 4vw, 2.6rem);
            font-weight: 900;
            letter-spacing: -0.055em;
            line-height: 1.05;
        }}

        .subtitle {{
            color: #AFC0CE;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-top: 0.5rem;
        }}

        .live-label {{
            color: {COLORS["green"]};
            font-size: 0.77rem;
            font-weight: 850;
            letter-spacing: 0.09em;
            margin-top: 0.65rem;
        }}

        .status-row {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin: 0.9rem 0;
        }}

        .pill {{
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            font-size: 0.77rem;
            font-weight: 850;
        }}

        .normal {{
            background: {COLORS["green"]};
            color: #04130D;
        }}

        .warning {{
            background: {COLORS["yellow"]};
            color: #261B03;
        }}

        .danger {{
            background: {COLORS["red"]};
            color: #28050B;
        }}

        .muted {{
            color: {COLORS["muted"]};
            font-size: 0.8rem;
        }}

        .metric-card {{
            background:
                linear-gradient(
                    150deg,
                    {COLORS["surface_2"]},
                    {COLORS["surface"]}
                );
            border: 1px solid {COLORS["border"]};
            border-radius: 18px;
            padding: 0.9rem 1rem;
            min-height: 112px;
            box-shadow: 0 9px 27px rgba(0,0,0,0.15);
        }}

        .metric-label {{
            color: {COLORS["muted"]};
            font-size: 0.7rem;
            font-weight: 750;
            letter-spacing: 0.055em;
            text-transform: uppercase;
        }}

        .metric-value {{
            color: {COLORS["white"]};
            font-size: 1.8rem;
            font-weight: 900;
            margin-top: 0.35rem;
        }}

        .metric-note {{
            color: #AFC0CE;
            font-size: 0.74rem;
            margin-top: 0.22rem;
        }}

        .section-title {{
            color: {COLORS["white"]};
            font-size: 1.2rem;
            font-weight: 850;
            margin-top: 1.25rem;
            margin-bottom: 0.65rem;
        }}

        .alert-box {{
            border-radius: 17px;
            padding: 0.9rem 1rem;
            margin: 0.8rem 0 1rem;
            border-left: 6px solid {COLORS["red"]};
            background:
                linear-gradient(
                    110deg,
                    rgba(255,92,119,0.16),
                    rgba(255,92,119,0.025)
                );
        }}

        .alert-title {{
            color: #FF9AA9;
            font-size: 0.93rem;
            font-weight: 850;
        }}

        .alert-text {{
            color: #E0EAF1;
            font-size: 0.82rem;
            line-height: 1.5;
            margin-top: 0.3rem;
        }}

        .footer {{
            color: #6F8498;
            text-align: center;
            font-size: 0.73rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid {COLORS["border"]};
        }}

        @media (max-width: 700px) {{
            .block-container {{
                padding-left: 0.6rem;
                padding-right: 0.6rem;
            }}

            .header {{
                border-radius: 17px;
                padding: 0.95rem;
            }}

            .subtitle {{
                font-size: 0.78rem;
            }}

            .metric-card {{
                min-height: 100px;
            }}

            .metric-value {{
                font-size: 1.42rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_csv(
    path_string: str,
    modified_time: float,
) -> pd.DataFrame:
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce",
        )

    return data


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_mtime = (
        PREDICTIONS_PATH.stat().st_mtime
        if PREDICTIONS_PATH.exists()
        else 0.0
    )

    alert_mtime = (
        ALERTS_PATH.stat().st_mtime
        if ALERTS_PATH.exists()
        else 0.0
    )

    predictions = read_csv(
        str(PREDICTIONS_PATH),
        prediction_mtime,
    )

    alerts = read_csv(
        str(ALERTS_PATH),
        alert_mtime,
    )

    return predictions, alerts


def number(
    value: float,
    digits: int = 1,
) -> str:
    return f"{value:,.{digits}f}"


def time_text(
    value: Optional[pd.Timestamp],
) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return value.strftime(
        "%Y-%m-%d %H:%M"
    )


def file_time(
    path: Path,
) -> str:
    if not path.exists():
        return "Missing"

    modified = pd.Timestamp.fromtimestamp(
        path.stat().st_mtime
    )

    return time_text(modified)


def get_status(
    alerts: pd.DataFrame,
) -> tuple[str, str, str]:
    if alerts.empty:
        return (
            "NORMAL",
            "No persistent alerts",
            "normal",
        )

    danger_count = int(
        (
            alerts["severity"] == "DANGER"
        ).sum()
    )

    warning_count = int(
        (
            alerts["severity"] == "WARNING"
        ).sum()
    )

    if danger_count > 0:
        return (
            "DANGER",
            f"{danger_count} danger alert(s)",
            "danger",
        )

    if warning_count > 0:
        return (
            "WARNING",
            f"{warning_count} warning alert(s)",
            "warning",
        )

    return (
        "NORMAL",
        "No persistent alerts",
        "normal",
    )


def render_sidebar(
    predictions: pd.DataFrame,
    alerts: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    int,
]:
    with st.sidebar:
        st.markdown(
            "## ☀️ SolarPulse"
        )

        st.caption(
            "PV predictive-maintenance control panel"
        )

        st.success(
            "● LIVE CONTROL CENTER"
        )

        status, status_text, _ = get_status(
            alerts
        )

        sorted_predictions = predictions.copy()

        if (
            not sorted_predictions.empty
            and "timestamp" in sorted_predictions.columns
        ):
            sorted_predictions = sorted_predictions.sort_values(
                "timestamp"
            )

        if (
            not sorted_predictions.empty
            and "risk_score" in sorted_predictions.columns
        ):
            risk_values = pd.to_numeric(
                sorted_predictions["risk_score"],
                errors="coerce",
            ).dropna()
        else:
            risk_values = pd.Series(
                dtype="float64"
            )

        latest_risk = (
            float(risk_values.iloc[-1])
            if not risk_values.empty
            else 0.0
        )

        maximum_risk = (
            float(risk_values.max())
            if not risk_values.empty
            else 0.0
        )

        average_risk = (
            float(risk_values.mean())
            if not risk_values.empty
            else 0.0
        )

        danger_count = int(
            (
                alerts["severity"] == "DANGER"
            ).sum()
        ) if (
            not alerts.empty
            and "severity" in alerts.columns
        ) else 0

        warning_count = int(
            (
                alerts["severity"] == "WARNING"
            ).sum()
        ) if (
            not alerts.empty
            and "severity" in alerts.columns
        ) else 0

        health = max(
            0.0,
            min(
                100.0,
                100.0
                - latest_risk
                - danger_count * 2.0,
            ),
        )

        st.markdown(
            "### System status"
        )

        st.metric(
            "Current state",
            status,
            status_text,
        )

        st.progress(
            int(round(health)),
            text=(
                f"Operational health: "
                f"{health:.1f}%"
            ),
        )

        st.markdown(
            "### Live summary"
        )

        summary_left, summary_right = st.columns(
            2
        )

        with summary_left:
            st.metric(
                "Rows",
                f"{len(predictions):,}",
            )

            st.metric(
                "Warning",
                f"{warning_count:,}",
            )

        with summary_right:
            st.metric(
                "Latest risk",
                f"{latest_risk:.1f}",
            )

            st.metric(
                "Danger",
                f"{danger_count:,}",
            )

        st.caption(
            f"Average risk: {average_risk:.1f}"
        )

        st.caption(
            f"Maximum risk: {maximum_risk:.1f}"
        )

        st.divider()

        st.markdown(
            "### Filters"
        )

        filtered_predictions = predictions.copy()
        filtered_alerts = alerts.copy()

        if (
            not predictions.empty
            and "timestamp" in predictions.columns
        ):
            timestamps = predictions[
                "timestamp"
            ].dropna()

            if not timestamps.empty:
                min_date = timestamps.min().date()
                max_date = timestamps.max().date()

                selected_dates = st.date_input(
                    "Analysis period",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )

                if (
                    isinstance(selected_dates, tuple)
                    and len(selected_dates) == 2
                ):
                    start_time = pd.Timestamp(
                        selected_dates[0]
                    )

                    end_time = (
                        pd.Timestamp(
                            selected_dates[1]
                        )
                        + pd.Timedelta(days=1)
                    )

                    filtered_predictions = (
                        filtered_predictions[
                            (
                                filtered_predictions[
                                    "timestamp"
                                ]
                                >= start_time
                            )
                            & (
                                filtered_predictions[
                                    "timestamp"
                                ]
                                < end_time
                            )
                        ]
                    )

                    if (
                        not filtered_alerts.empty
                        and "timestamp"
                        in filtered_alerts.columns
                    ):
                        filtered_alerts = (
                            filtered_alerts[
                                (
                                    filtered_alerts[
                                        "timestamp"
                                    ]
                                    >= start_time
                                )
                                & (
                                    filtered_alerts[
                                        "timestamp"
                                    ]
                                    < end_time
                                )
                            ]
                        )

        if (
            not filtered_alerts.empty
            and "severity" in filtered_alerts.columns
        ):
            severity_values = sorted(
                filtered_alerts["severity"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_severities = st.multiselect(
                "Alert severity",
                severity_values,
                default=severity_values,
            )

            filtered_alerts = filtered_alerts[
                filtered_alerts["severity"].isin(
                    selected_severities
                )
            ]

        if (
            not filtered_predictions.empty
            and "system_state"
            in filtered_predictions.columns
        ):
            state_values = sorted(
                filtered_predictions["system_state"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_states = st.multiselect(
                "System state",
                state_values,
                default=state_values,
            )

            filtered_predictions = filtered_predictions[
                filtered_predictions["system_state"].isin(
                    selected_states
                )
            ]

        refresh_seconds = st.slider(
            "Auto-refresh interval",
            min_value=0,
            max_value=300,
            value=30,
            step=10,
            help=(
                "Set 0 to disable automatic refresh."
            ),
        )

        if st.button(
            "Refresh now",
            use_container_width=True,
        ):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        st.markdown(
            "### Status legend"
        )

        st.success(
            "NORMAL · Stable operation"
        )

        st.warning(
            "WARNING · Needs attention"
        )

        st.error(
            "DANGER · Requires review"
        )

        st.divider()

        st.markdown(
            "### Data sources"
        )

        st.caption(
            f"Predictions: {PREDICTIONS_PATH.name}"
        )

        st.caption(
            f"Updated: {file_time(PREDICTIONS_PATH)}"
        )

        st.caption(
            f"Alerts: {ALERTS_PATH.name}"
        )

        st.caption(
            f"Updated: {file_time(ALERTS_PATH)}"
        )

        st.markdown(
            """
            <div class="sidebar-footer">
                Auto-refresh only reloads the dashboard view.
                It does not execute the prediction pipeline.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return (
        filtered_predictions,
        filtered_alerts,
        refresh_seconds,
    )


def chart_style(
    figure: go.Figure,
) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,23,37,0.78)",
        font={
            "family": "Arial",
            "color": COLORS["white"],
        },
        margin={
            "l": 12,
            "r": 12,
            "t": 58,
            "b": 12,
        },
        hovermode="x unified",
        legend={
            "orientation": "h",
            "y": 1.02,
            "x": 0,
        },
    )

    figure.update_xaxes(
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["grid"],
    )

    figure.update_yaxes(
        gridcolor=COLORS["grid"],
        zerolinecolor=COLORS["grid"],
    )

    return figure


def create_power_chart(
    data: pd.DataFrame,
) -> Optional[go.Figure]:
    required = {
        "timestamp",
        "power_dc",
        "predicted_power",
    }

    if not required.issubset(data.columns):
        return None

    chart_data = data.sort_values(
        "timestamp"
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data["timestamp"],
            y=chart_data["power_dc"],
            mode="lines",
            name="Actual power",
            line={
                "color": COLORS["green"],
                "width": 2.8,
            },
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data["timestamp"],
            y=chart_data["predicted_power"],
            mode="lines",
            name="Predicted power",
            line={
                "color": COLORS["blue"],
                "width": 2,
                "dash": "dot",
            },
        )
    )

    figure.update_layout(
        title="Live power production",
        xaxis_title="Time",
        yaxis_title="Power",
    )

    return chart_style(figure)


def create_risk_chart(
    data: pd.DataFrame,
) -> Optional[go.Figure]:
    required = {
        "timestamp",
        "risk_score",
    }

    if not required.issubset(data.columns):
        return None

    chart_data = data.sort_values(
        "timestamp"
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data["timestamp"],
            y=chart_data["risk_score"],
            mode="lines",
            name="Risk",
            line={
                "color": COLORS["red"],
                "width": 2.6,
            },
            fill="tozeroy",
            fillcolor="rgba(255,92,119,0.10)",
        )
    )

    figure.add_hline(
        y=40,
        line_dash="dash",
        line_color=COLORS["yellow"],
        annotation_text="WARNING 40",
    )

    figure.add_hline(
        y=70,
        line_dash="dash",
        line_color=COLORS["red"],
        annotation_text="DANGER 70",
    )

    figure.update_layout(
        title="Live risk trend",
        xaxis_title="Time",
        yaxis_title="Risk score",
        yaxis_range=[0, 100],
    )

    return chart_style(figure)


def create_alert_chart(
    alerts: pd.DataFrame,
) -> Optional[go.Figure]:
    required = {
        "timestamp",
        "severity",
        "current_risk_score",
    }

    if alerts.empty or not required.issubset(
        alerts.columns
    ):
        return None

    figure = go.Figure()

    for severity, color in [
        ("WARNING", COLORS["yellow"]),
        ("DANGER", COLORS["red"]),
    ]:
        subset = alerts[
            alerts["severity"] == severity
        ]

        if subset.empty:
            continue

        figure.add_trace(
            go.Scatter(
                x=subset["timestamp"],
                y=subset["current_risk_score"],
                mode="markers",
                name=severity,
                marker={
                    "size": 12,
                    "color": color,
                    "line": {
                        "width": 1,
                        "color": COLORS["white"],
                    },
                },
                text=subset["predicted_fault"],
            )
        )

    figure.update_layout(
        title="Alert event timeline",
        xaxis_title="Time",
        yaxis_title="Current risk",
        yaxis_range=[0, 100],
    )

    return chart_style(figure)


def create_gauge(
    value: float,
    title: str,
    color: str,
    suffix: str = "",
) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(0, min(100, value)),
            number={
                "suffix": suffix,
                "font": {
                    "color": COLORS["white"],
                    "size": 34,
                },
            },
            title={
                "text": title,
                "font": {
                    "color": COLORS["muted"],
                    "size": 16,
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": COLORS["muted"],
                },
                "bar": {
                    "color": color,
                },
                "bgcolor": "#14263A",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, 40],
                        "color": "rgba(0,212,168,0.16)",
                    },
                    {
                        "range": [40, 70],
                        "color": "rgba(246,183,60,0.16)",
                    },
                    {
                        "range": [70, 100],
                        "color": "rgba(255,92,119,0.16)",
                    },
                ],
                "threshold": {
                    "line": {
                        "color": COLORS["white"],
                        "width": 3,
                    },
                    "thickness": 0.8,
                    "value": 70,
                },
            },
        )
    )

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=270,
        margin={
            "l": 18,
            "r": 18,
            "t": 42,
            "b": 5,
        },
    )

    return figure


def render_panel_array(
    status: str,
    power: float,
    risk: float,
) -> None:
    panel_color = {
        "NORMAL": "#168C86",
        "WARNING": "#C38A24",
        "DANGER": "#B63A54",
    }.get(
        status,
        "#168C86",
    )

    panel_glow = {
        "NORMAL": "rgba(0,212,168,0.80)",
        "WARNING": "rgba(246,183,60,0.85)",
        "DANGER": "rgba(255,92,119,0.88)",
    }.get(
        status,
        "rgba(0,212,168,0.80)",
    )

    panels = ""

    for panel_number in range(1, 13):
        panels += f"""
        <div class="pv-panel pv-{panel_number}">
            <div class="cell-grid"></div>
            <div class="panel-reflection"></div>
            <div class="panel-number">
                PV-{panel_number:02d}
            </div>
        </div>
        """

    html = f"""
    <div style="
        width:100%;
        height:430px;
        position:relative;
        overflow:hidden;
        border-radius:24px;
        background:
            linear-gradient(
                180deg,
                #06152A 0%,
                #0B3551 58%,
                #06131F 100%
            );
        border:1px solid #23455C;
        box-shadow:0 18px 50px rgba(0,0,0,0.26);
        font-family:Arial,sans-serif;
    ">
        <style>
        .scene {{
            position:absolute;
            inset:0;
            overflow:hidden;
        }}

        .stars {{
            position:absolute;
            inset:0;
            opacity:0.42;
            background-image:
                radial-gradient(
                    circle at 11% 18%,
                    #FFFFFF 0 1px,
                    transparent 2px
                ),
                radial-gradient(
                    circle at 31% 12%,
                    #FFFFFF 0 1px,
                    transparent 2px
                ),
                radial-gradient(
                    circle at 67% 22%,
                    #FFFFFF 0 1px,
                    transparent 2px
                ),
                radial-gradient(
                    circle at 86% 14%,
                    #FFFFFF 0 1px,
                    transparent 2px
                );
        }}

        .sun {{
            position:absolute;
            top:34px;
            right:10%;
            width:82px;
            height:82px;
            border-radius:50%;
            background:
                radial-gradient(
                    circle at 34% 28%,
                    #FFF8B0,
                    #FFD45E 42%,
                    #FF9D1F 75%,
                    #ED5C0C
                );
            box-shadow:
                0 0 22px #FFD45E,
                0 0 60px rgba(255,191,60,0.86),
                0 0 110px rgba(255,163,48,0.34);
            animation:sunPulse 3s ease-in-out infinite;
        }}

        .sun-ring {{
            position:absolute;
            top:19px;
            right:calc(10% - 15px);
            width:112px;
            height:112px;
            border:2px solid rgba(255,216,93,0.30);
            border-radius:50%;
            animation:orbit 9s linear infinite;
        }}

        .sun-ring-two {{
            position:absolute;
            top:6px;
            right:calc(10% - 28px);
            width:138px;
            height:138px;
            border:2px solid rgba(255,216,93,0.14);
            border-radius:50%;
            animation:orbit 13s linear infinite reverse;
        }}

        .ray {{
            position:absolute;
            right:calc(10% + 39px);
            top:69px;
            width:4px;
            height:25px;
            border-radius:5px;
            background:#FFD45E;
            transform-origin:2px 67px;
            animation:rotateRay 5s linear infinite;
        }}

        .ray:nth-child(3) {{
            transform:rotate(45deg);
        }}

        .ray:nth-child(4) {{
            transform:rotate(90deg);
        }}

        .ray:nth-child(5) {{
            transform:rotate(135deg);
        }}

        .ray:nth-child(6) {{
            transform:rotate(180deg);
        }}

        .ray:nth-child(7) {{
            transform:rotate(225deg);
        }}

        .ray:nth-child(8) {{
            transform:rotate(270deg);
        }}

        .ray:nth-child(9) {{
            transform:rotate(315deg);
        }}

        .cloud {{
            position:absolute;
            width:92px;
            height:25px;
            border-radius:30px;
            background:rgba(190,222,236,0.13);
            animation:cloudMove 18s linear infinite;
        }}

        .cloud::before,
        .cloud::after {{
            content:"";
            position:absolute;
            bottom:3px;
            border-radius:50%;
            background:rgba(190,222,236,0.13);
        }}

        .cloud::before {{
            width:34px;
            height:34px;
            left:15px;
        }}

        .cloud::after {{
            width:45px;
            height:45px;
            left:43px;
        }}

        .cloud-one {{
            top:70px;
            left:-120px;
        }}

        .cloud-two {{
            top:122px;
            left:-280px;
            transform:scale(0.7);
            animation-delay:7s;
        }}

        .light-beam {{
            position:absolute;
            top:91px;
            right:15%;
            width:56%;
            height:220px;
            transform:skewX(-25deg);
            background:
                linear-gradient(
                    125deg,
                    transparent,
                    rgba(255,218,93,0.15),
                    transparent
                );
            animation:beamMove 4s ease-in-out infinite;
        }}

        .ground {{
            position:absolute;
            left:-6%;
            right:-6%;
            bottom:-96px;
            height:190px;
            border-radius:50% 50% 0 0;
            background:
                radial-gradient(
                    ellipse at center,
                    #0A2A31,
                    #06151F 68%
                );
        }}

        .array {{
            position:absolute;
            left:6%;
            bottom:80px;
            width:58%;
            height:200px;
            transform:skewY(-8deg) rotate(-1deg);
        }}

        .pv-panel {{
            position:absolute;
            width:22%;
            height:43%;
            border:3px solid #B4D5E2;
            border-radius:6px;
            overflow:hidden;
            background:
                linear-gradient(
                    135deg,
                    {panel_color},
                    #07345D
                );
            box-shadow:
                0 9px 19px rgba(0,0,0,0.35),
                0 0 17px {panel_glow};
            animation:panelPulse 3.5s ease-in-out infinite;
        }}

        .pv-panel:nth-child(2n) {{
            animation-delay:0.45s;
        }}

        .pv-panel:nth-child(3n) {{
            animation-delay:0.9s;
        }}

        .cell-grid {{
            position:absolute;
            inset:0;
            background:
                repeating-linear-gradient(
                    90deg,
                    transparent 0 18px,
                    rgba(180,231,255,0.58) 19px 21px
                ),
                repeating-linear-gradient(
                    0deg,
                    transparent 0 18px,
                    rgba(180,231,255,0.58) 19px 21px
                );
            opacity:0.74;
        }}

        .panel-reflection {{
            position:absolute;
            inset:-40% -20%;
            background:
                linear-gradient(
                    115deg,
                    transparent 38%,
                    rgba(205,245,255,0.78) 49%,
                    transparent 62%
                );
            transform:translateX(-100%);
            animation:panelShine 4.2s ease-in-out infinite;
        }}

        .panel-number {{
            position:absolute;
            right:4px;
            bottom:3px;
            color:rgba(238,251,255,0.82);
            font-size:8px;
            font-weight:800;
        }}

        .pv-1 {{
            left:0;
            top:0;
        }}

        .pv-2 {{
            left:25%;
            top:0;
        }}

        .pv-3 {{
            left:50%;
            top:0;
        }}

        .pv-4 {{
            left:75%;
            top:0;
        }}

        .pv-5 {{
            left:0;
            top:49%;
        }}

        .pv-6 {{
            left:25%;
            top:49%;
        }}

        .pv-7 {{
            left:50%;
            top:49%;
        }}

        .pv-8 {{
            left:75%;
            top:49%;
        }}

        .pv-9 {{
            left:0;
            top:98%;
        }}

        .pv-10 {{
            left:25%;
            top:98%;
        }}

        .pv-11 {{
            left:50%;
            top:98%;
        }}

        .pv-12 {{
            left:75%;
            top:98%;
        }}

        .energy-bus {{
            position:absolute;
            left:63%;
            bottom:172px;
            width:18%;
            height:4px;
            border-radius:5px;
            background:
                linear-gradient(
                    90deg,
                    transparent,
                    #FFD45E,
                    #00D4A8,
                    transparent
                );
            box-shadow:0 0 14px #00D4A8;
            animation:flow 1.8s linear infinite;
        }}

        .energy-dot {{
            position:absolute;
            left:64%;
            bottom:167px;
            width:11px;
            height:11px;
            border-radius:50%;
            background:#00D4A8;
            box-shadow:0 0 18px #00D4A8;
            animation:dotMove 2.4s linear infinite;
        }}

        .inverter {{
            position:absolute;
            right:8%;
            bottom:80px;
            width:104px;
            height:110px;
            border:2px solid #69889A;
            border-radius:12px;
            background:#10283A;
            box-shadow:0 10px 25px rgba(0,0,0,0.42);
        }}

        .inverter-screen {{
            position:absolute;
            left:12px;
            right:12px;
            top:13px;
            height:29px;
            border-radius:5px;
            background:#06151F;
            color:{panel_glow};
            text-align:center;
            font-size:10px;
            font-weight:850;
            padding-top:9px;
            box-shadow:inset 0 0 12px {panel_glow};
        }}

        .inverter-led {{
            position:absolute;
            left:17px;
            bottom:17px;
            width:9px;
            height:9px;
            border-radius:50%;
            background:{panel_glow};
            box-shadow:0 0 14px {panel_glow};
            animation:ledBlink 1.35s ease-in-out infinite;
        }}

        .inverter-label {{
            position:absolute;
            right:14px;
            bottom:15px;
            color:#A5BBC7;
            font-size:10px;
        }}

        .ac-output {{
            position:absolute;
            right:6%;
            bottom:48px;
            width:45px;
            height:3px;
            background:#4EA1FF;
            box-shadow:0 0 12px #4EA1FF;
            animation:acFlow 1.4s linear infinite;
        }}

        .scene-label {{
            position:absolute;
            top:16px;
            left:18px;
            color:#EAF7FF;
            font-size:13px;
            font-weight:850;
            letter-spacing:0.08em;
        }}

        .scene-data {{
            position:absolute;
            top:41px;
            left:18px;
            color:#A8C9D8;
            font-size:11px;
        }}

        .scene-status {{
            position:absolute;
            top:18px;
            right:18px;
            color:{panel_glow};
            font-size:11px;
            font-weight:850;
        }}

        @keyframes sunPulse {{
            0%,100% {{
                transform:scale(1);
            }}
            50% {{
                transform:scale(1.08);
            }}
        }}

        @keyframes orbit {{
            from {{
                transform:rotate(0);
            }}
            to {{
                transform:rotate(360deg);
            }}
        }}

        @keyframes rotateRay {{
            from {{
                transform:rotate(0);
            }}
            to {{
                transform:rotate(360deg);
            }}
        }}

        @keyframes cloudMove {{
            from {{
                transform:translateX(0);
            }}
            to {{
                transform:translateX(900px);
            }}
        }}

        @keyframes beamMove {{
            0%,100% {{
                opacity:0.25;
                transform:skewX(-25deg)
                translateX(-20px);
            }}
            50% {{
                opacity:0.85;
                transform:skewX(-25deg)
                translateX(24px);
            }}
        }}

        @keyframes panelPulse {{
            0%,100% {{
                filter:brightness(0.92);
            }}
            50% {{
                filter:brightness(1.12);
            }}
        }}

        @keyframes panelShine {{
            0% {{
                transform:translateX(-100%);
            }}
            55%,100% {{
                transform:translateX(100%);
            }}
        }}

        @keyframes flow {{
            0% {{
                opacity:0.3;
                background-position:0 0;
            }}
            50% {{
                opacity:1;
            }}
            100% {{
                opacity:0.45;
                background-position:180px 0;
            }}
        }}

        @keyframes dotMove {{
            0% {{
                transform:translateX(0);
                opacity:0;
            }}
            15% {{
                opacity:1;
            }}
            90% {{
                opacity:1;
            }}
            100% {{
                transform:translateX(170px);
                opacity:0;
            }}
        }}

        @keyframes acFlow {{
            0% {{
                opacity:0.25;
                transform:translateX(-15px);
            }}
            50% {{
                opacity:1;
            }}
            100% {{
                opacity:0.25;
                transform:translateX(20px);
            }}
        }}

        @keyframes ledBlink {{
            0%,100% {{
                opacity:0.35;
            }}
            50% {{
                opacity:1;
            }}
        }}

        @media (max-width:700px) {{
            .array {{
                left:4%;
                width:65%;
                height:160px;
                bottom:84px;
            }}

            .inverter {{
                right:4%;
                width:78px;
                height:90px;
            }}

            .inverter-screen {{
                font-size:8px;
                padding-top:8px;
            }}

            .inverter-label {{
                font-size:8px;
                right:8px;
            }}

            .sun {{
                right:9%;
                width:60px;
                height:60px;
            }}

            .ray {{
                right:calc(9% + 28px);
                top:52px;
            }}

            .scene-status {{
                font-size:9px;
            }}
        }}
        </style>

        <div class="scene">
            <div class="stars"></div>

            <div class="scene-label">
                PV ARRAY LIVE VIEW
            </div>

            <div class="scene-data">
                Power: {power:.1f} · Risk: {risk:.1f}
            </div>

            <div class="scene-status">
                STATUS: {status}
            </div>

            <div class="sun"></div>
            <div class="sun-ring"></div>
            <div class="sun-ring-two"></div>

            <div class="ray"></div>
            <div class="ray"></div>
            <div class="ray"></div>
            <div class="ray"></div>
            <div class="ray"></div>
            <div class="ray"></div>
            <div class="ray"></div>

            <div class="cloud cloud-one"></div>
            <div class="cloud cloud-two"></div>

            <div class="light-beam"></div>
            <div class="ground"></div>

            <div class="array">
                {panels}
            </div>

            <div class="energy-bus"></div>
            <div class="energy-dot"></div>

            <div class="inverter">
                <div class="inverter-screen">
                    {status}
                </div>
                <div class="inverter-led"></div>
                <div class="inverter-label">
                    INVERTER
                </div>
            </div>

            <div class="ac-output"></div>
        </div>
    </div>
    """

    if hasattr(st, "html"):
        st.html(
            html,
            unsafe_allow_javascript=True,
        )
    else:
        st.components.v1.html(
            html,
            height=450,
            scrolling=False,
        )


def render_metric(
    column,
    label: str,
    value: str,
    note: str,
    color: str,
) -> None:
    with column:
        st.markdown(
            f"""
            <div class="metric-card"
                 style="border-top:3px solid {color};">
                <div class="metric-label">
                    {label}
                </div>
                <div class="metric-value">
                    {value}
                </div>
                <div class="metric-note">
                    {note}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_alert(
    alerts: pd.DataFrame,
) -> None:
    if alerts.empty:
        st.success(
            "No persistent alerts match the current filters."
        )
        return

    latest = alerts.sort_values(
        "timestamp"
    ).iloc[-1]

    severity = str(
        latest.get("severity", "WARNING")
    )

    fault = str(
        latest.get(
            "predicted_fault",
            "unknown",
        )
    )

    state = str(
        latest.get(
            "current_system_state",
            "unknown",
        )
    )

    risk = float(
        latest.get(
            "current_risk_score",
            0,
        )
    )

    action = str(
        latest.get(
            "recommended_action",
            "Continue monitoring.",
        )
    )

    st.markdown(
        f"""
        <div class="alert-box">
            <div class="alert-title">
                Latest persistent {severity} event
            </div>
            <div class="alert-text">
                Fault: {fault} · State: {state} ·
                Current risk: {risk:.1f}
            </div>
            <div class="alert-text">
                Recommended action: {action}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            SolarPulse PV Command Center ·
            Live photovoltaic monitoring ·
            Model output is decision support, not a confirmed
            physical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()

    predictions, alerts = load_data()

    if predictions.empty:
        st.error(
            "Prediction file is missing or empty: "
            f"{PREDICTIONS_PATH}"
        )
        st.stop()

    (
        filtered_predictions,
        filtered_alerts,
        refresh_seconds,
    ) = render_sidebar(
        predictions,
        alerts,
    )

    refresh_count = 0

    if refresh_seconds > 0:
        refresh_count = st_autorefresh(
            interval=refresh_seconds * 1000,
            key="solar_pv_sidebar_refresh",
        )

    status, status_detail, status_class = get_status(
        filtered_alerts
    )

    latest = (
        filtered_predictions.sort_values(
            "timestamp"
        ).iloc[-1]
        if not filtered_predictions.empty
        else None
    )

    latest_power = (
        float(
            latest.get(
                "power_dc",
                0,
            )
        )
        if latest is not None
        else 0.0
    )

    latest_risk = (
        float(
            latest.get(
                "risk_score",
                0,
            )
        )
        if latest is not None
        else 0.0
    )

    latest_time = (
        latest.get(
            "timestamp",
            None,
        )
        if latest is not None
        else None
    )

    average_risk = (
        float(
            filtered_predictions["risk_score"].mean()
        )
        if (
            not filtered_predictions.empty
            and "risk_score"
            in filtered_predictions.columns
        )
        else 0.0
    )

    warning_count = int(
        (
            filtered_alerts["severity"]
            == "WARNING"
        ).sum()
    ) if not filtered_alerts.empty else 0

    danger_count = int(
        (
            filtered_alerts["severity"]
            == "DANGER"
        ).sum()
    ) if not filtered_alerts.empty else 0

    st.markdown(
        """
        <div class="header">
            <div class="title">
                ☀️ SolarPulse PV Command Center
            </div>
            <div class="subtitle">
                Multi-panel photovoltaic monitoring,
                predictive maintenance, and live energy flow
            </div>
            <div class="live-label">
                ● LIVE PV ARRAY OPERATIONS
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-row">
            <span class="pill {status_class}">
                SYSTEM {status}
            </span>
            <span class="muted">
                {status_detail}
            </span>
            <span class="muted">
                · Data: {time_text(latest_time)}
            </span>
            <span class="muted">
                · Refresh: {refresh_count}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_alert(
        filtered_alerts
    )

    st.markdown(
        '<div class="section-title">'
        "Animated multi-panel PV array"
        "</div>",
        unsafe_allow_html=True,
    )

    render_panel_array(
        status=status,
        power=latest_power,
        risk=latest_risk,
    )

    st.markdown(
        '<div class="section-title">'
        "Live indicators"
        "</div>",
        unsafe_allow_html=True,
    )

    metric_1, metric_2, metric_3, metric_4, metric_5 = (
        st.columns(5, gap="medium")
    )

    render_metric(
        metric_1,
        "Observations",
        f"{len(filtered_predictions):,}",
        "Filtered records",
        COLORS["blue"],
    )

    render_metric(
        metric_2,
        "Warnings",
        f"{warning_count:,}",
        "Persistent warnings",
        COLORS["yellow"],
    )

    render_metric(
        metric_3,
        "Danger",
        f"{danger_count:,}",
        "Persistent danger",
        COLORS["red"],
    )

    render_metric(
        metric_4,
        "Average risk",
        number(average_risk),
        "Filtered average",
        COLORS["green"],
    )

    render_metric(
        metric_5,
        "Latest power",
        number(latest_power),
        "Current DC output",
        COLORS["blue"],
    )

    st.divider()

    gauge_left, gauge_right = st.columns(
        2,
        gap="large",
    )

    with gauge_left:
        st.plotly_chart(
            create_gauge(
                latest_risk,
                "Current risk score",
                COLORS["red"],
            ),
            use_container_width=True,
        )

    health_score = max(
        0,
        min(
            100,
            100
            - latest_risk
            - danger_count * 2,
        ),
    )

    health_color = (
        COLORS["green"]
        if health_score >= 75
        else COLORS["yellow"]
        if health_score >= 45
        else COLORS["red"]
    )

    with gauge_right:
        st.plotly_chart(
            create_gauge(
                health_score,
                "Operational health",
                health_color,
                "%",
            ),
            use_container_width=True,
        )

    chart_left, chart_right = st.columns(
        [1.4, 1],
        gap="large",
    )

    with chart_left:
        figure = create_power_chart(
            filtered_predictions
        )

        if figure is not None:
            st.plotly_chart(
                figure,
                use_container_width=True,
            )

    with chart_right:
        figure = create_risk_chart(
            filtered_predictions
        )

        if figure is not None:
            st.plotly_chart(
                figure,
                use_container_width=True,
            )

    figure = create_alert_chart(
        filtered_alerts
    )

    if figure is not None:
        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    snapshot_tab, alert_tab, data_tab = st.tabs(
        [
            "Operational snapshot",
            "Alert center",
            "Data explorer",
        ]
    )

    with snapshot_tab:
        latest_state = (
            str(
                latest.get(
                    "system_state",
                    "N/A",
                )
            )
            if latest is not None
            else "N/A"
        )

        latest_fault = (
            str(
                latest.get(
                    "predicted_fault",
                    "N/A",
                )
            )
            if latest is not None
            else "N/A"
        )

        latest_temperature = (
            float(
                latest.get(
                    "panel_temperature",
                    0,
                )
            )
            if latest is not None
            else 0.0
        )

        latest_irradiance = (
            float(
                latest.get(
                    "irradiance",
                    0,
                )
            )
            if latest is not None
            else 0.0
        )

        snap_1, snap_2, snap_3, snap_4 = st.columns(
            4,
            gap="medium",
        )

        with snap_1:
            st.metric(
                "Current state",
                latest_state,
            )

        with snap_2:
            st.metric(
                "Predicted fault",
                latest_fault,
            )

        with snap_3:
            st.metric(
                "Panel temperature",
                number(latest_temperature),
            )

        with snap_4:
            st.metric(
                "Irradiance",
                number(latest_irradiance),
            )

    with alert_tab:
        if filtered_alerts.empty:
            st.success(
                "No persistent alert matches the filters."
            )
        else:
            columns = [
                column
                for column in [
                    "timestamp",
                    "severity",
                    "current_system_state",
                    "predicted_fault",
                    "current_risk_score",
                    "warning_count_in_window",
                    "danger_count_in_window",
                    "panel_temperature",
                    "irradiance",
                    "classifier_confidence",
                    "recommended_action",
                ]
                if column in filtered_alerts.columns
            ]

            st.dataframe(
                filtered_alerts.sort_values(
                    "timestamp",
                    ascending=False,
                )[columns],
                use_container_width=True,
                hide_index=True,
                height=500,
            )

            csv_data = filtered_alerts.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download alerts CSV",
                data=csv_data,
                file_name="solarpulse_alerts.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with data_tab:
        if filtered_predictions.empty:
            st.info(
                "No prediction records match the filters."
            )
        else:
            search = st.text_input(
                "Search prediction data",
                placeholder=(
                    "Search state, fault, timestamp, or value..."
                ),
            )

            table_data = filtered_predictions.copy()

            if search.strip():
                searchable = table_data.astype(
                    str
                ).apply(
                    lambda column: column.str.contains(
                        search,
                        case=False,
                        na=False,
                    )
                )

                table_data = table_data[
                    searchable.any(axis=1)
                ]

            max_rows = max(
                10,
                min(1000, len(table_data)),
            )

            row_count = st.slider(
                "Rows to display",
                min_value=10,
                max_value=max_rows,
                value=min(50, max_rows),
                step=10,
            )

            st.dataframe(
                table_data.tail(row_count),
                use_container_width=True,
                hide_index=True,
                height=520,
            )

            csv_data = table_data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download predictions CSV",
                data=csv_data,
                file_name="solarpulse_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.caption(
        "Prediction source: solar_predictions.csv · "
        "Alert source: alerts.csv · "
        f"Latest record: {time_text(latest_time)}"
    )

    render_footer()


if __name__ == "__main__":
    main()