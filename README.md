# SolarSense AI

## Intelligent Solar Panel Monitoring, Performance Analysis, Fault Detection, and Predictive Maintenance System

### Project Overview

SolarSense AI is an interdisciplinary engineering project for monitoring photovoltaic systems, analyzing solar-panel performance, detecting abnormal production patterns, and supporting predictive-maintenance decisions.

The project combines:

- Mechanical engineering
- Photovoltaic-system analysis
- Electronics and sensors
- Internet of Things
- Python programming
- Data analysis
- Machine learning
- Streamlit dashboard
- Telegram notifications

### Problem Definition

Photovoltaic power production changes because of:

- Solar irradiance
- Panel temperature
- Ambient temperature
- Clouds
- Partial shading
- Panel soiling
- Loose connections
- Sensor faults
- Panel disconnection
- Wiring problems

SolarSense AI will use electrical and environmental data to estimate expected power, compare it with measured power, detect abnormal behavior, classify possible faults, and present the results in a local dashboard.

The model output is an engineering estimate. It does not guarantee physical failure prediction and does not replace electrical protection or professional inspection.

### Initial Architecture

```text
Synthetic Data or ESP32 Sensors
            |
            v
Local CSV or Local API
            |
            v
Data Validation
            |
            v
Feature Engineering
            |
            v
Machine-Learning Models
            |
            v
Status and Alert Logic
            |
            +------> Telegram Notifications
            |
            v
Streamlit Dashboard
```

### Safe Prototype Scope

The first prototype will use synthetic data and low-voltage laboratory-scale hardware.

Possible safe sources include:

- A small low-voltage photovoltaic panel
- A laboratory DC power supply used as a panel simulator
- A 5 V or 12 V educational solar kit
- Properly rated voltage and current sensors

Never connect a high-voltage photovoltaic panel directly to an ESP32 or Arduino.

The ESP32 ADC cannot accept arbitrary voltage. Voltage dividers must be designed according to voltage rating, power dissipation, resistor values, and ADC limits. Current sensors must have an appropriate measurement range. Fuses, current limiting, suitable wiring, and safe connections are required.

Electronics must first be tested without connecting a real solar panel.

### Repository Structure

```text
solar-predictive-maintenance/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── models/
├── results/
│   ├── charts/
│   ├── reports/
│   └── exports/
├── dashboard/
│   └── pages/
├── src/
│   └── utils/
├── notebooks/
├── tests/
├── .streamlit/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

### Current Stage

Stage 1 creates the repository structure and Git configuration.

Future stages will add:

1. Synthetic photovoltaic-data generation
2. Data validation
3. Physical calculations
4. Feature engineering
5. Anomaly detection
6. Fault classification
7. Expected-power prediction
8. Alert persistence logic
9. Streamlit dashboard
10. Telegram reporting
11. Optional ESP32 integration
12. Optional local FastAPI backend

### Limitations

The initial dataset will be simulated. Synthetic ground-truth labels do not prove field performance. Real hardware requires calibration, protection, suitable sensor ranges, and electrical-safety verification.

Machine-learning predictions must be interpreted as possible abnormal patterns or estimated risks.

### Future Improvements

- Calibrated irradiance measurement
- ESP32 data acquisition
- Local API communication
- Model monitoring
- Historical field-data validation
- Hardware-in-the-loop testing
- Weather-data integration
- Inverter-data integration
- Maintenance reports

### License

An MIT License may be suitable for an educational portfolio, subject to the author's intended use.