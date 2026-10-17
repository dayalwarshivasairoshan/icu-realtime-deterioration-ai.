# Real-Time AI for Critical Decisions

Real-time clinical AI that detects patient deterioration early (sepsis, cardiac arrest, respiratory failure) and reduces alarm fatigue, designed for **IBM LinuxONE (Linux on Z)**.

> **IBM Z Datathon submission** | Challenge: *Real-Time AI for Critical Decisions*

## The problem
Delayed recognition of deterioration increases mortality, while frequent false or low-priority alarms cause alarm fatigue and missed critical events.

## Our solution
A streaming pipeline that scores every patient continuously and raises prioritized, de-duplicated alerts:

1. **Capture** - vitals (HR, BP, SpO2, RR, temperature) and lab values (lactate) arrive per patient each minute.
2. **Featurize** - rolling 10-minute window: latest value, mean and slope per signal.
3. **Predict** - a gradient-boosting model outputs a deterioration risk score (0 to 1).
4. **Prioritize** - alerts need a *sustained* risk (3 consecutive minutes), are tiered WARNING (>= 0.70) / CRITICAL (>= 0.90), escalate on worsening, and use a 30-minute cooldown to avoid repeats.
5. **Act** - a browser dashboard shows risk trends and the alert feed.

## Run it (about 1 minute)
```bash
pip install -r requirements.txt
python run_demo.py        # trains the model, simulates a 40-patient ward, writes dashboard.html
python -m pytest -q tests # unit tests
```
Then open `dashboard.html` in any browser.

### Sample output (synthetic data, seed 2026)
| Metric | Result |
|---|---|
| Fixed-threshold monitor alarms | 604 |
| AI prioritized alerts | 56 (about 91% fewer interruptions) |
| Deteriorating patients detected | 12 of 12 |
| Median warning before critical | about 42 minutes |
| False alerts (stable patient or before onset) | 4 |

## How it uses IBM Z / LinuxONE
| Capability | Role in the design |
|---|---|
| **Telum on-chip AI accelerator** | Millisecond inference next to the data via ONNX Runtime / Snap ML, with no external GPU round trips |
| **Pervasive encryption + Secure Execution** | Confidential computing for patient data, supporting HIPAA / GDPR compliance |
| **99.999% availability** | Critical alerts are never missed |
| **Scale and workload isolation** | Thousands of concurrent patient streams with consistent latency |
| **Red Hat OpenShift on LinuxONE** | Cloud-native deployment of the stream, inference and alert services |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the target architecture and the path from this prototype to LinuxONE.

## Honest scope
- This prototype runs on **synthetic data** and **standard CPU Python**. It has not been benchmarked on Telum hardware.
- The model score (AUC near 1.0) is inflated because training and test data come from the same simulator. Real clinical validation is future work.
- The stream is simulated in-process; Kafka, Db2/PostgreSQL and EHR (HL7 / FHIR) integration are part of the target architecture, not this repo.
- **Not a medical device.** For research and demonstration only.

## Repo layout
```
src/simulator.py     synthetic bedside monitors (with motion-artifact false alarms)
src/features.py      rolling-window features
src/risk_model.py    offline training and scoring
src/alert_engine.py  tiered, de-duplicated alerts + naive threshold baseline
src/pipeline.py      end-to-end run and metrics
run_demo.py          entry point, writes dashboard.html
tests/               unit tests
docs/                architecture
```

## Roadmap
Validate on public clinical datasets, add explainable risk factors, integrate EHR via FHIR, pilot in a ward, and add federated learning across facilities.

## Team
Add your team name and members here.

## License
MIT
