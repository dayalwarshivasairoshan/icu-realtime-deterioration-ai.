"""End-to-end run: stream -> features -> risk model -> alert engine -> metrics + dashboard data."""
import numpy as np
from .alert_engine import AlertEngine, NaiveMonitor
from .features import FeatureWindow
from .risk_model import risk, train
from .simulator import RAMP_MINUTES, make_cohort, stream


def run(n_patients: int = 40, horizon: int = 240, seed: int = 2026):
    model, auc = train()
    patients = make_cohort(n_patients, seed, frac_sick=0.4, horizon=horizon)
    by_id = {p.pid: p for p in patients}
    windows = {p.pid: FeatureWindow() for p in patients}
    engine, naive = AlertEngine(), NaiveMonitor()
    series = {p.pid: {"risk": [], "spo2": [], "hr": []} for p in patients}
    alerts, naive_alarms = [], {p.pid: 0 for p in patients}

    for t, p, r in stream(patients, horizon, seed + 1):
        f = windows[p.pid].push(r)
        score = risk(model, f)
        series[p.pid]["risk"].append(round(score, 3))
        series[p.pid]["spo2"].append(round(float(r[2]), 1))
        series[p.pid]["hr"].append(round(float(r[0]), 1))
        if naive.update(p.pid, r):
            naive_alarms[p.pid] += 1
        a = engine.update(t, p.pid, score)
        if a:
            alerts.append(a)

    sick = [p for p in patients if p.condition != "stable"]
    first_alert = {}
    for a in alerts:
        first_alert.setdefault(a.pid, a.t)
    detected = [p for p in sick if p.pid in first_alert and first_alert[p.pid] >= p.onset]
    leads = [p.onset + RAMP_MINUTES - first_alert[p.pid] for p in detected]
    # an alert is "false" if the patient is stable or has not yet begun to deteriorate
    false_alerts = sum(1 for a in alerts if by_id[a.pid].condition == "stable" or a.t < by_id[a.pid].onset)
    summary = {
        "patients": n_patients, "minutes": horizon, "model_auc": round(auc, 3),
        "naive_alarms": int(sum(naive_alarms.values())),
        "ai_alerts": len(alerts), "ai_false_alerts": int(false_alerts),
        "deteriorating": len(sick), "detected": len(detected),
        "median_lead_min": float(np.median(leads)) if leads else 0.0,
    }
    pts = [{"id": p.pid, "condition": p.condition, "onset": p.onset, "naive": naive_alarms[p.pid],
            **series[p.pid]} for p in patients]
    return summary, pts, [a.__dict__ for a in alerts]
