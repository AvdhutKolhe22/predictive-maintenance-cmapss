# NASA C-MAPSS Predictive Maintenance System

An end-to-end deep learning predictive-maintenance system built using the NASA C-MAPSS FD001 turbofan engine dataset.

## Overview

The system combines two CNN-BiLSTM models:

- **V2:** Remaining Useful Life (RUL) estimation
- **V3:** Failure-risk prediction

Their outputs are combined to generate an actionable maintenance decision:

**NORMAL → MONITOR → WARNING → CRITICAL**

## System Architecture

Engine Sensor Data → 30-Cycle × 16-Feature Window → CNN-BiLSTM Models → V2 RUL + V3 Failure Risk → Hybrid Maintenance Decision → Maintenance Recommendation

## Dataset

NASA C-MAPSS FD001 turbofan engine dataset. The system uses 16 selected features and processes 30 consecutive operating cycles for each prediction.

## Final Model Performance

### V2 — RUL Model

| Metric | Result |
|---|---:|
| MAE | 28.9519 |
| RMSE | 39.2669 |
| R² | 0.4825 |

### V3 — Failure-Risk Model

| Metric | Result |
|---|---:|
| PR-AUC | 0.9183 |
| ROC-AUC | 0.9966 |
| Precision | 0.7944 |
| Recall | 0.8494 |
| F1 | 0.8210 |

### Engine-Level Warning Performance

- Warning coverage: **88%**
- Mean warning lead time: **30.91 cycles**
- Missed failures: **3 / 25**

## Hybrid Maintenance System

The final system combines **V2 Predicted RUL + V3 Failure Probability** to determine the maintenance status.

| Status | Recommendation |
|---|---|
| NORMAL | Continue normal operation |
| MONITOR | Increase monitoring |
| WARNING | Plan maintenance |
| CRITICAL | Schedule maintenance immediately |

## Example — Engine 20

- Predicted RUL: **22.01 cycles**
- Failure probability: **93.37%**
- Status: **CRITICAL**
- Recommendation: **Schedule maintenance immediately**

## Interactive Dashboard

The project includes a Streamlit dashboard for real-engine inference.

```bash
streamlit run app.py

