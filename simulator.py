"""Synthetic bedside-monitor simulator.

Generates one reading per patient per simulated minute. Some patients stay stable,
others progress toward sepsis, cardiac arrest or respiratory failure over ~60 minutes.
Motion artifacts (short, harmless spikes) are injected to mimic real false alarms.
All data is SYNTHETIC. No real patient data is used.
"""
from dataclasses import dataclass
import numpy as np

VITALS = ["hr", "sbp", "spo2", "rr", "temp", "lactate"]
CONDITIONS = ["stable", "sepsis", "cardiac", "respiratory"]
RAMP_MINUTES = 60  # minutes from onset until the patient is critical

# change in each vital at full progression: (hr, sbp, spo2, rr, temp, lactate)
SHIFTS = {
    "sepsis":      np.array([40, -35, -3, 10, 1.8, 3.0]),
    "cardiac":     np.array([55, -45, -6, 6, 0.0, 2.0]),
    "respiratory": np.array([25, -5, -12, 14, 0.2, 1.0]),
}
NOISE = np.array([3.0, 4.0, 0.8, 1.2, 0.1, 0.15])


@dataclass
class Patient:
    pid: str
    condition: str
    onset: int
    baseline: np.ndarray

    def progress(self, t: int) -> float:
        if self.condition == "stable" or t < self.onset:
            return 0.0
        return min(1.0, (t - self.onset) / RAMP_MINUTES)

    def reading(self, t: int, rng: np.random.Generator) -> np.ndarray:
        v = self.baseline + rng.normal(0, 1, 6) * NOISE
        p = self.progress(t)
        if p > 0:
            v = v + SHIFTS[self.condition] * p
        if rng.random() < 0.03:  # motion artifact: one-tick spike, not a real event
            v[2] -= 8.0
            v[0] += 25.0
        v[2] = min(v[2], 100.0)
        return v


def make_cohort(n: int, seed: int, frac_sick: float = 0.4, horizon: int = 240):
    rng = np.random.default_rng(seed)
    patients = []
    for i in range(n):
        base = np.array([
            rng.normal(78, 9), rng.normal(122, 10), rng.normal(96.5, 1.1),
            rng.normal(15, 2), rng.normal(36.8, 0.25), rng.normal(1.1, 0.25),
        ])
        if rng.random() < frac_sick:
            cond = str(rng.choice(CONDITIONS[1:]))
            onset = int(rng.integers(40, horizon - RAMP_MINUTES - 10))
        else:
            cond, onset = "stable", -1
        patients.append(Patient(f"P{i + 1:03d}", cond, onset, base))
    return patients


def stream(patients, horizon: int, seed: int):
    """Yield (t, patient, reading) in time order, like a Kafka topic would."""
    rng = np.random.default_rng(seed)
    for t in range(horizon):
        for p in patients:
            yield t, p, p.reading(t, rng)
