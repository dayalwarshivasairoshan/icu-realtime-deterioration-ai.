import numpy as np
from src.alert_engine import AlertEngine, NaiveMonitor
from src.features import FeatureWindow
from src.simulator import make_cohort


def test_features_shape():
    w = FeatureWindow()
    for _ in range(12):
        f = w.push(np.ones(6))
    assert f.shape == (18,)


def test_single_spike_does_not_alert():
    e = AlertEngine()
    out = [e.update(t, "P1", 0.95 if t == 5 else 0.1) for t in range(20)]
    assert all(a is None for a in out)


def test_sustained_risk_alerts_once_per_cooldown():
    e = AlertEngine()
    alerts = [a for t in range(25) if (a := e.update(t, "P1", 0.8))]
    assert len(alerts) == 1 and alerts[0].tier == "WARNING"


def test_escalation_to_critical():
    e = AlertEngine()
    tiers = [a.tier for t in range(12) if (a := e.update(t, "P1", 0.8 if t < 5 else 0.95))]
    assert tiers == ["WARNING", "CRITICAL"]


def test_naive_monitor_fires_on_breach_edge():
    n = NaiveMonitor()
    ok, bad = np.array([75, 120, 97, 15, 36.8, 1.0]), np.array([75, 120, 90, 15, 36.8, 1.0])
    assert [n.update("P1", r) for r in (ok, bad, bad, ok, bad)] == [False, True, False, False, True]


def test_cohort_reproducible():
    a, b = make_cohort(10, 1), make_cohort(10, 1)
    assert [p.condition for p in a] == [p.condition for p in b]
