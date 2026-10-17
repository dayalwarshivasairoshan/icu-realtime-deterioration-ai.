# Architecture

## Target deployment on IBM LinuxONE

```
Bedside monitors / lab systems
        |  (HL7 / device feeds)
        v
Apache Kafka  -------------------------------+
        |                                     |
        v                                     v
Feature service  ->  Inference service     Data store
(rolling windows)    (ONNX Runtime /       (Db2 or PostgreSQL)
                      Snap ML on Telum)
        |
        v
Alert engine (tiering, sustained-risk rule, cooldown)
        |
        v
Clinician dashboard / mobile notifications

All services run as containers on Red Hat OpenShift on IBM LinuxONE,
protected by pervasive encryption and Secure Execution.
```

## What this repository implements today
| Stage | In this repo | Target on LinuxONE |
|---|---|---|
| Ingestion | In-process simulator | Apache Kafka topics per ward |
| Features | `src/features.py` | Same logic as a stream-processing service |
| Model | scikit-learn gradient boosting | Export to ONNX, serve with ONNX Runtime / Snap ML on Telum |
| Alerts | `src/alert_engine.py` | Same logic, plus paging and mobile push |
| Storage | JSON in `dashboard.html` | Db2 / PostgreSQL with pervasive encryption |
| UI | Static HTML dashboard | Hospital dashboard |

## Design choices
- **Sustained-risk rule (3 minutes)** filters one-minute motion artifacts, a main cause of false alarms.
- **Tiers and escalation** put the sickest patients first.
- **Cooldown** prevents repeated alerts for the same patient and tier.
- **Inference beside the data** is the main reason to run on Z: low latency and no data leaving the secured platform.
